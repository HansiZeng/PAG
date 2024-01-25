import argparse
import os 
import ujson

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default=None, type=str)
    parser.add_argument("--out_dir", default=None, type=str)

    args = parser.parse_args()

    return args 

if __name__ == "__main__":
    args = get_args()

    with open(os.path.join(args.out_dir, "docid_to_smtid.json")) as fin:
        docid_to_smtids = ujson.load(fin)

    with open(os.path.join(args.model_dir, "extended_token_checkpoint/added_tokens.json")) as fin:
        tokenids_map = ujson.load(fin)

    docid_to_tokenids = {}    
    for docid, smtids in docid_to_smtids.items():
        tokenids = []
        for i, smtid in enumerate(smtids):
            key = f"<docid_{i}_{smtid}>"
            tokenids.append(tokenids_map[key])
        docid_to_tokenids[docid] = tokenids 

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    assert "aq_smtid" in args.out_dir, args.out_dir 
    with open(os.path.join(args.out_dir, "docid_to_tokenids.json"), "w") as fout:
        ujson.dump(docid_to_tokenids, fout)