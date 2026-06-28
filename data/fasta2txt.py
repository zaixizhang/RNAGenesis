# modified from https://github.com/facebookresearch/esm/blob/main/esm/data.py

import os


def convert(fasta_file, out_file, min_seq_length = None,  max_seq_length = None, replace=False):
    """
        min_seq_length (included), max_seq_length (excluded)
    """
    count = 0
    if min_seq_length is not None:
        print(f'sequence below sequence length: {min_seq_length} will be removed.')
    if max_seq_length is not None:
        print(f'sequence exceding max sequence length: {max_seq_length} will be removed.')

    with open (out_file, 'w') as out_f_handle:
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf, count
            if cur_seq_label is None:
                return
            seq = "".join(buf)
            seq = seq.upper()
            if replace:
                seq = seq.replace("T", "U") # For (nc)RNA, replace T with U
            remove = False 
            if min_seq_length is not None and len(seq) < min_seq_length:
                remove = True
            if max_seq_length is not None and len(seq) >= max_seq_length: 
                remove = True
            if not remove:
                out_f_handle.write(seq)
                out_f_handle.write("\n")
                count += 1
            cur_seq_label = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()
    print(f"dataset size: {count}, saved in {out_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=int, default=None, help="minimum sequence length")
    parser.add_argument("--max", type=int, default=None, help="maximum sequence length")
    parser.add_argument("--replace", action="store_true", help="whether to replace T with U")
    parser.add_argument("--fasta_file", type=str, default=None, help="fasta file name")
    parser.add_argument("--folder", type=str, default="/scratch/gpfs/yy1325/data", help="folder path to fasta file")

    args = parser.parse_args()
    min, max, replace, fasta_file = args.min, args.max, args.replace, args.fasta_file
    folder = args.folder
    if fasta_file is None:
        fasta_files = [f for f in os.listdir(folder) if f.endswith('.fasta') or f.endswith('.fa')]
    else:
        fasta_files = [fasta_file]
    for fasta_file in fasta_files:
        convert(
            os.path.join(folder, fasta_file), os.path.join(folder, f'{os.path.splitext(fasta_file)[0]}_min{min}max{max}.txt'), 
            min_seq_length = min, max_seq_length = max, replace=replace
            )
        print("Done converting", fasta_file)