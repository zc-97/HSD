from HsdUtils.utils import get_model
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from torch.nn.init import xavier_uniform_, xavier_normal_


class HSD(SequentialRecommender):

    def __init__(self, config, dataset):
        super(HSD, self).__init__(config, dataset)

        # load parameters info
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.our_ae_drop_out = config['our_ae_drop_out']
        self.our_att_drop_out = config['our_att_drop_out']
        self.n_users = dataset.num(self.USER_ID)    # Compatible with the latest version of RecBole

        self.tau = 100
        self.filter_drop_rate = 0.0

        self.user_embedding = nn.Embedding(self.n_users, self.hidden_size, padding_idx=0)

        self.LayerNorm = nn.BatchNorm1d(self.hidden_size)
        self.emb_dropout = nn.Dropout(self.our_ae_drop_out)

        # 加入bi-LSTM
        self.rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            bidirectional=True,
            batch_first=True
        )

        self.conv = nn.Conv2d(self.max_seq_length, self.max_seq_length, (1, 2))
        self.seq_level_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 2, bias=False),
            nn.Sigmoid()
        )

        self.read_out = AttnReadout(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.hidden_size,
            session_len=self.max_seq_length,
            batch_norm=True,
            feat_drop=self.our_att_drop_out,
            activation=nn.PReLU(self.hidden_size),
        )

        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=1)
        self.binary_softmax = nn.Softmax(dim=-1)

        self.loss_fuc = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

        # 初始化sub_model
        self.sub_model = get_model(config['sub_model'])(config, dataset).to(config['device'])
        self.sub_model_name = config['sub_model']
        self.item_embedding = self.sub_model.item_embedding

        if config['load_pre_train_emb'] is not None and config['load_pre_train_emb']:
            checkpoint_file = config['pre_train_model_dict'][config['dataset']][config['sub_model']]
            checkpoint = torch.load(checkpoint_file)
            if config['sub_model'] == 'DSAN':
                embedding_weight = checkpoint['state_dict']['embedding.weight']
                self.item_embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
            else:
                item_embedding_weight = checkpoint['state_dict']['item_embedding.weight']
                self.item_embedding = nn.Embedding.from_pretrained(item_embedding_weight, freeze=False)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def method_name(self, generated_seq, generated_seq_emb):
        row_indexes, col_id = torch.where(generated_seq.gt(0))
        row_flag = row_indexes[0]
        index_flag = -1
        col_index = []
        for row_index in row_indexes:
            if row_index == row_flag:
                index_flag += 1
                col_index.append(index_flag)
            else:
                index_flag = 0
                col_index.append(index_flag)
                row_flag = row_index
        col_index = torch.tensor(col_index, device=self.device)
        denoising_seq = torch.zeros_like(generated_seq)
        denoising_seq_emb = torch.zeros_like(generated_seq_emb)
        denoising_seq[row_indexes, col_index] = generated_seq[row_indexes, col_id]
        denoising_seq_emb[row_indexes, col_index, :] = generated_seq_emb[row_indexes, col_id, :]
        return denoising_seq, denoising_seq_emb

    def generate_pos_seq(self, item_seq, item_seq_len, item_level_score, seq_level_score, mask):
        item_emb = self.item_embedding(item_seq)
        mask = mask.squeeze()

        # Todo 可调整
        item_level_gumbel_softmax_rst = F.gumbel_softmax(item_level_score, tau=self.tau, hard=True)
        seq_level_gumbel_softmax_rst = F.gumbel_softmax(seq_level_score, tau=self.tau, hard=True)

        item_level_denoising_seq_flag = item_level_gumbel_softmax_rst[:, :, 1] * mask
        seq_level_denoising_seq_flag = seq_level_gumbel_softmax_rst[:, :, 1] * mask

        noisy_flag = item_level_denoising_seq_flag * seq_level_denoising_seq_flag

        pos_flag = (1 - noisy_flag) * mask

        pos_seq_emb = item_emb * pos_flag.unsqueeze(-1)

        pos_seq = item_seq * pos_flag
        pos_seq[pos_seq != pos_seq] = 0

        pos_seq_len = torch.sum(pos_flag, dim=-1)

        # 如果序列为0 stamp会报错，因此这里将序列长度为0 的保留第一个
        pos_seq_len[pos_seq_len.eq(0)] = 1

        # TODO: [1,2,3,4] -> [1,0,3,4] or [1,3,4]
        clean_seq_percent = torch.sum(pos_seq_len, dim=0) / item_seq_len.sum() * 100
        denoising_seq, denoising_seq_emb = self.method_name(pos_seq, pos_seq_emb)
        pos_seq = denoising_seq
        pos_seq_emb = denoising_seq_emb
        neg_seq_len = torch.squeeze(item_seq_len)

        return pos_seq, pos_seq_emb, pos_seq_len.long(), item_seq, neg_seq_len, clean_seq_percent

    def seq_level_consistency(self, item_seq, item_seq_len, mask, train_flag=True):
        item_seq_emb_ori = self.item_embedding(item_seq)

        item_seq_emb = self.emb_dropout(item_seq_emb_ori) * mask

        encoder_item_seq_emb_bi_direction, (encoder_hidden, mm_) = self.rnn(item_seq_emb)

        '将最后一个时刻的hidden state两个方向加起来'
        encoder_hidden = (encoder_hidden[0] + encoder_hidden[1]).squeeze()

        # torch.Size([2048, 200, 64])
        rnn1_hidden = int(encoder_item_seq_emb_bi_direction.shape[-1] / 2)
        encoder_item_seq_emb = encoder_item_seq_emb_bi_direction[:, :, :rnn1_hidden] + \
                               encoder_item_seq_emb_bi_direction[:, :, rnn1_hidden:]

        # TODO: 尝试随即删除部分item来输入到双向LSTM中
        encoder_item_seq_emb = encoder_item_seq_emb * mask
        decoder_item_seq_emb_bi_direction, _ = self.rnn(encoder_item_seq_emb)
        rnn2_hidden = int(decoder_item_seq_emb_bi_direction.shape[-1] / 2)
        decoder_item_seq_emb = decoder_item_seq_emb_bi_direction[:, :, :rnn2_hidden] + \
                               decoder_item_seq_emb_bi_direction[:, :, rnn2_hidden:]

        element_wise_reconstruction_loss = 0

        if train_flag:
            loss_fct = nn.MSELoss(reduction='none')
            element_wise_reconstruction_loss = loss_fct(decoder_item_seq_emb * mask, item_seq_emb_ori * mask).sum(
                -1).sum(
                -1) / item_seq_len.squeeze()

        concat_shuffled_and_origin = torch.stack((decoder_item_seq_emb, item_seq_emb_ori), dim=-1)  # [B len 2xh]
        concat_shuffled_and_origin = self.conv(concat_shuffled_and_origin)  # [B len h 1]
        concat_shuffled_and_origin = torch.squeeze(concat_shuffled_and_origin)  # [B len h]
        concat_shuffled_and_origin = self.emb_dropout(concat_shuffled_and_origin)  # [B len h]
        concat_shuffled_and_origin = nn.ReLU(inplace=True)(concat_shuffled_and_origin)  # [B len h]

        reconstruct_score = self.seq_level_mlp(concat_shuffled_and_origin).squeeze()  # [B len 2]

        reconstruct_score = reconstruct_score * mask
        return element_wise_reconstruction_loss, reconstruct_score, encoder_item_seq_emb, encoder_hidden

    def item_level_consistency(self, item_seq_emb, target_embedding, mask):
        item_level_score, item_level_long_term_representation, item_level_seq_emb = self.read_out(item_seq_emb,
                                                                                                  target_embedding,
                                                                                                  mask)

        return item_level_score, item_level_long_term_representation, item_level_seq_emb

    def loss_filter(self, user, item_seq, item_seq_len, interaction, mask, train_flag):
        item_seq_emb = self.item_embedding(item_seq)  # [B, L, 1, H]
        item_seq_emb = (item_seq_emb * mask).unsqueeze(-2)
        user_emb = self.user_embedding(user).unsqueeze(-2).unsqueeze(-1)  # [B, 1, H, 1]

        pos_score = torch.matmul(item_seq_emb, user_emb)  # [B, L, 1]

        filter_drop_rate = self.filter_drop_rate

        if train_flag:
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items = neg_items.unsqueeze(-1).expand(item_seq.shape)
            neg_items_emb = self.item_embedding(neg_items)  # [B, L, 1, H]
            neg_items_emb = (neg_items_emb * mask).unsqueeze(-2)
            neg_score = torch.matmul(neg_items_emb, user_emb)  # [B, L, 1]
        else:
            neg_score = torch.zeros_like(pos_score)
            filter_drop_rate = 0.2

        loss = -torch.log(1e-10 + torch.sigmoid(pos_score - neg_score)).squeeze(-1).squeeze(-1)  # [B, L, 1]
        loss = loss * mask.squeeze(-1)
        loss_sorted, ind_sorted = torch.sort(loss, descending=False, dim=-1)
        num_remember = (filter_drop_rate * item_seq_len).squeeze(-1).long()

        loss_filter_flag = torch.zeros_like(item_seq)

        for index, filtered_item_num in enumerate(num_remember):
            loss_index = ind_sorted[index][-filtered_item_num:]
            loss_filter_flag[index][loss_index] = 1
            if filter_drop_rate != 0:
                loss[index][loss_index] *= 0
        loss_filter_flag = loss_filter_flag * mask.squeeze(-1)
        return loss, loss_filter_flag

    def forward(self, interaction, train_flag=True):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN].unsqueeze(1)
        if train_flag:
            target_item = interaction[self.ITEM_ID].unsqueeze(1)
        else:
            '如果是验证和测试的时候，由于不能提前预知下一项，因此将训练集的最后一项视作target'
            target_item = item_seq.gather(1, item_seq_len - 1)
        user = interaction[self.USER_ID]

        # calculate the mask
        mask = torch.ones(item_seq.shape, dtype=torch.float, device=item_seq.device) * item_seq.gt(0)

        # size = [batchSize, user_num_per_batch, 1]
        mask = mask.unsqueeze(2)

        item_seq_emb = self.item_embedding(item_seq)

        element_wise_reconstruction_loss, seq_level_score, seq_level_encoder_item_emb, seq_level_seq_emb = self.seq_level_consistency(
            item_seq=item_seq,
            item_seq_len=item_seq_len,
            mask=mask,
            train_flag=train_flag
        )

        if train_flag:
            target_embedding = self.item_embedding(target_item)
        else:
            target_embedding = self.user_embedding(user.unsqueeze(-1))
        item_level_score, item_level_long_term_representation, item_level_seq_emb = self.item_level_consistency(
            item_seq_emb=item_seq_emb,
            target_embedding=target_embedding,
            mask=mask)

        '将item- and seq-level 的embedding作为 emb-level denoising 从而增强data-level denoising 效果'
        embedding_level_denoising_emb = item_level_seq_emb + seq_level_seq_emb

        loss, loss_filter_flag = self.loss_filter(user, item_seq, item_seq_len, interaction, mask, train_flag)

        pos_seq, pos_seq_emb, pos_seq_len, neg_seq, neg_seq_len, clean_seq_percent = self.generate_pos_seq(
            item_seq=item_seq,
            item_seq_len=item_seq_len,
            item_level_score=item_level_score,
            seq_level_score=seq_level_score,
            mask=mask,
        )

        return element_wise_reconstruction_loss, pos_seq, pos_seq_emb, pos_seq_len, neg_seq, neg_seq_len, embedding_level_denoising_emb, clean_seq_percent, loss.sum(-1).mean()

    def calculate_loss_curriculum(self, interaction, drop_rate, tau):
        self.tau = tau
        self.filter_drop_rate = 0.2 - drop_rate

        element_wise_reconstruction_loss, pos_seq, pos_seq_emb, pos_seq_len, neg_seq, neg_seq_len, embedding_level_denoising_emb, clean_seq_percent, loss_filter_loss = self.forward(interaction)
        'positive seq                  --- A'
        'embedding_level_denoising_emb --- B'
        'negative seq                  --- C'
        'min/max D(AB)-D(AC)'

        user = interaction[self.USER_ID] if self.sub_model_name == 'Caser' else None
        all_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token

        # using the denoisied embedding calculate predict loss
        sub_model_output = self.sub_model_forward(pos_seq, pos_seq_emb, pos_seq_len, user)
        seq_representation, delete_index = self.denoising_seq_gather(pos_seq, sub_model_output)
        scores = torch.matmul(seq_representation, all_items_emb.transpose(0, 1))  # [B, item_num]

        ind_update = self.cal_curriculum_batch_id(drop_rate, element_wise_reconstruction_loss)
        element_wise_reconstruction_loss_curriculum = element_wise_reconstruction_loss[ind_update]
        L_rec = element_wise_reconstruction_loss_curriculum.mean()

        target_item = interaction[self.ITEM_ID].unsqueeze(1)
        target_item = target_item.squeeze()
        generated_seq_loss = self.loss_fuc(scores[ind_update], target_item[ind_update])

        total_loss = L_rec + generated_seq_loss + loss_filter_loss
        return total_loss, clean_seq_percent, L_rec, generated_seq_loss

    def cal_curriculum_batch_id(self, drop_rate, element_wise_reconstruction_loss):
        loss_sorted, ind_sorted = torch.sort(element_wise_reconstruction_loss, descending=False)
        remember_rate = 1 - drop_rate
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]
        return ind_update

    def predict(self, interaction):
        user = interaction[self.USER_ID] if self.sub_model_name == 'Caser' else None
        element_wise_reconstruction_loss, pos_seq, generated_seq, denoising_seq_len, temp1_, temp2_, embedding_level_denoising_emb, clean_seq_percent, loss = self.forward(
            interaction,
            train_flag=False)
        seq_output = self.sub_model_forward(generated_seq, denoising_seq_len, user)

        seq_output, _ = self.denoising_seq_gather(generated_seq, seq_output)

        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def denoising_seq_gather(self, generated_seq, seq_output):
        generated_item_seq_len = torch.sum(generated_seq.gt(0), 1)

        # 算出长度为0的则是全为噪声项的序列，这里先记录其index
        delete_index = torch.where(generated_item_seq_len.eq(0))[0]
        # 将index减一 ，防止数组越上界
        generated_item_seq_len = generated_item_seq_len - 1
        # 将index为-1 的项置为0，防止数组越下界
        generated_item_seq_len = generated_item_seq_len * generated_item_seq_len.gt(0)
        if self.sub_model_name in ['SRGNN', 'GCSAN', 'Caser', 'NARM', 'DSAN', 'STAMP']:
            seq_output = seq_output
        elif self.sub_model_name == 'fmlp':
            seq_output = seq_output[:, -1, :]  # delete masked token
        else:
            seq_output = self.gather_indexes(seq_output, generated_item_seq_len)  # [B H]
        return seq_output, delete_index

    def sub_model_forward(self, generated_seq, pos_seq_emb, denoising_seq_len, user):
        if self.sub_model_name == 'BERT4Rec':
            seq_output = self.sub_model.forward_denoising(generated_seq, pos_seq_emb)
        elif self.sub_model_name == 'GRU4Rec':
            seq_output = self.sub_model.forward_denoising(generated_seq, pos_seq_emb)
        elif self.sub_model_name == 'SASRec':
            seq_output = self.sub_model.forward_denoising(generated_seq, pos_seq_emb)
        elif self.sub_model_name == 'Caser':
            seq_output = self.sub_model.forward_denoising(user, generated_seq, pos_seq_emb)
        elif self.sub_model_name == 'NARM':
            seq_output = self.sub_model.forward_denoising(generated_seq, denoising_seq_len, pos_seq_emb)
        elif self.sub_model_name == 'DSAN':
            seq_output, _ = self.sub_model.forward(generated_seq)
        elif self.sub_model_name == 'fmlp':
            seq_output = self.sub_model.forward(generated_seq)
        elif self.sub_model_name == 'STAMP':
            seq_output = self.sub_model.forward_denoising(generated_seq, denoising_seq_len, pos_seq_emb)
        else:
            raise ValueError(f'Sub_model [{self.sub_model_name}] not support.')
        return seq_output

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID] if self.sub_model_name == 'Caser' else None
        element_wise_reconstruction_loss, pos_seq, pos_seq_emb, denoising_seq_len, _, _, embedding_level_denoising_emb, pre, loss = self.forward(
            interaction,
            train_flag=False)
        seq_output = self.sub_model_forward(pos_seq, pos_seq_emb, denoising_seq_len, user)

        seq_output, _ = self.denoising_seq_gather(pos_seq, seq_output)

        test_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores, pre


class AttnReadout(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            session_len,
            batch_norm=True,
            feat_drop=0.0,
            activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(session_len) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 2, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim else None
        )
        self.mlp_n_ls = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feat, last_nodes, mask):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = feat * mask
        feat = self.feat_drop(feat)
        feat = feat * mask
        feat_u = self.fc_u(feat)
        feat_u = feat_u * mask
        feat_v = self.fc_v(last_nodes)  # (batch_size * embedding_size)

        e = self.fc_e(F.tanh(feat_u + feat_v)) * mask
        e = self.sigmoid(e)

        short_term = last_nodes.squeeze()

        e0, rst = self.get_long_term(e, feat, mask)
        fuse_long_short = torch.cat((rst, short_term), dim=-1)
        item_level_seq_representation = self.mlp_n_ls(self.feat_drop(fuse_long_short))

        score = e.squeeze()
        return score, rst, item_level_seq_representation

    def get_long_term(self, e, feat, mask):
        'e是2维的分数，rst是long-term representation'
        mask1 = (mask - 1) * 2e32
        e0 = e[:, :, 1] + mask1.squeeze()
        beta = self.softmax(e0)
        feat_norm = feat * beta.unsqueeze(-1)
        feat_norm = feat_norm * mask
        rst = torch.sum(feat_norm, dim=1)
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return e0, rst
