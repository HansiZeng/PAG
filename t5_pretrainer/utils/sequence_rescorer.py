import torch 
import ujson 
import math
from copy import deepcopy



class BatchLexicalReScorer():
    def __init__(self, batch_prefixer, num_beams):
        ## smt_prefix --> docids 
        self.bid_to_rank = {}
        self.prefixer = batch_prefixer
        for bid, qid in enumerate(batch_prefixer._qids):
            self.bid_to_rank[bid] = {}
            for docid, score in batch_prefixer._qid_to_rankdata[str(qid)].items():
                self.bid_to_rank[bid][docid] = score
        
        self._num_beams = num_beams

    def _get_lex_score(self, batch_id, docids):
        max_score = 0.0 
        for docid in docids:
            max_score = max(self.bid_to_rank[batch_id].get(docid, 0.0), max_score)
        return max_score

    def __call__(self, input_ids, next_token_scores):
        """
        Args:
            input_ids: [bz*num_beams, cur_len]
            next_token_scores: [bz*num_beams, vocab_size]

        Returns:
            lex_inc_scores: [bz*num_beams, vocab_size]
        """
        _next_token_scores = next_token_scores.view(-1, self._num_beams, next_token_scores.shape[-1])
        lex_inc_scores = deepcopy(_next_token_scores)

        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            # beam_sent: [num_beams, cur_len]

            # 1 
            #for beam_id, sent in enumerate(beam_sent):
            #    next_ids = self.prefixer(batch_id, beam_sent[beam_id])
            #    for next_id in next_ids:
            #        full_sent = torch.cat((sent, torch.LongTensor([next_id]).to(sent.device)))
            #        allowed_docids = self.prefixer._get_docids(batch_id, full_sent)
            #        lex_score = self._get_lex_score(batch_id, allowed_docids)
            #        lex_inc_scores[batch_id, beam_id, next_id] += lex_score

            # 2  
            beam_ids, next_token_ids = (_next_token_scores[batch_id] > -1e2).nonzero(as_tuple=True)
            assert torch.sum(next_token_ids >= 32_000) == len(next_token_ids), (torch.sum(next_token_ids >= 32_000), len(next_token_ids))
            #print("next_ids: ", len(next_token_ids))
            #print(len(beam_ids), len(beam_sent[0]))
            for bid, next_id in zip(beam_ids, next_token_ids):
                bid = bid.item()
                full_sent = torch.cat((beam_sent[bid], next_id.view(-1)))
                allowed_docids = self.prefixer._get_docids(batch_id, full_sent)
                lex_score = self._get_lex_score(batch_id, allowed_docids)
                #print("next_ids_from prefixer: ", len(self.prefixer(batch_id, beam_sent[bid])), beam_sent[bid])

                lex_inc_scores[batch_id, bid, next_id] += lex_score

            # 3 
            # full_sents [N_nb, L], N_nb = len(beam_ids) = num_beams * avg(_) # first 2 tokens having 100 next_tokenids on average
            # 


        return lex_inc_scores

class BatchLexTmpReScorer():
    def __init__(self, batch_prefixer, num_beams):
        self.batch_prefixer = batch_prefixer
        self._num_beams = num_beams 

    def __call__(self, input_ids, scores):
        """
        Args:
            input_ids: [bz*num_beams, cur_len]
            scores: [bz*num_beams, vocab_size]

        Returns:
            lex_inc_scores: [bz*num_beams, vocab_size]
        """
        mask = torch.full_like(scores, 0.)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.size(-1))):
            for beam_id, sent in enumerate(beam_sent):
                token_ids, token_scores = self.batch_prefixer._get_tokenids_and_scores(batch_id, sent)
                #print("sent: ", len(sent), len(token_ids), torch.topk(token_scores, k=min(3, len(token_scores)), largest=True)[0],
                #      len(token_ids))
                mask[batch_id * self._num_beams + beam_id, token_ids] = token_scores

        return scores + mask
                
