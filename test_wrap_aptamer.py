import RNA
import numpy as np
import argparse

def get_gen_seqs(gen_seqs_path):
    gen_seqs = []
    with open(gen_seqs_path) as f:
        for line in f:
            # Remove '2' if present in the line
            if '2' in line:
                line = line.replace('2', '')
            line = line.strip()[1:]  # Remove the first character as needed
            # Add fixed fragments according to requirements
            line = 'TCCAGCACTCCACGCATAAC' + line + 'GTTATGCGTGCTACCGTGAA'
            gen_seqs.append(line)
    return gen_seqs

def get_pairs(structure):
    """
    Convert dot-bracket format structure to base pair set (0-based index).
    Example: structure = "((..))" returns {(0,5), (1,4)}
    """
    stack = []
    pairs = set()
    for i, ch in enumerate(structure):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pairs.add((j, i))
    return pairs

def compute_structure(seq):
    """
    Calculate RNA secondary structure and MFE.
    """
    structure, mfe = RNA.fold(seq)
    return structure, mfe

def compute_similarity(ref_structure, cand_structure, factor=2.0):
    """
    Calculate secondary structure similarity between reference and candidate structures (score out of 100).

    Steps:
      1. Calculate base pair overlap percentage using reference and candidate base pair sets.
      2. Use RNA.bp_distance to calculate basic pairing distance and convert to similarity score (higher score for smaller distance).
      3. Combine both scores (simple average) as final similarity.
    """
    ref_pairs = get_pairs(ref_structure)
    cand_pairs = get_pairs(cand_structure)
    if ref_pairs:
        pair_sim = len(ref_pairs & cand_pairs) / len(ref_pairs) * 100
    else:
        pair_sim = 100 if ref_structure == cand_structure else 0

    distance = RNA.bp_distance(cand_structure, ref_structure)
    distance_sim = max(0, 100 - distance * factor)
    overall = (pair_sim + distance_sim) / 2
    return overall

def batch_analysis(candidate_seqs, ref_seq, factor=2.0):
    """
    Perform batch calculations for candidate and reference sequences:
      1. Calculate reference sequence secondary structure and MFE (once)
      2. For each candidate sequence, calculate secondary structure, MFE, similarity to reference structure, and MFE difference
    """
    ref_structure, ref_mfe = compute_structure(ref_seq)
    results = []
    for seq in candidate_seqs:
        structure, mfe = compute_structure(seq)
        similarity = compute_similarity(ref_structure, structure, factor)
        mfe_diff = abs(mfe - ref_mfe)
        results.append({
            'seq': seq,
            'structure': structure,
            'mfe': mfe,
            'similarity': similarity,
            'mfe_diff': mfe_diff
        })
    return ref_structure, ref_mfe, results

def threshold_analysis(results, threshold_A, threshold_B):
    """
    Count separately:
      - Proportion of sequences with structure similarity >= threshold A (out of 100)
      - Proportion of sequences with MFE difference <= threshold B
    """
    total = len(results)
    count_similarity = sum(1 for r in results if r['similarity'] >= threshold_A)
    count_mfe = sum(1 for r in results if r['mfe_diff'] <= threshold_B)
    prop_similarity = count_similarity / total * 100 if total > 0 else 0
    prop_mfe = count_mfe / total * 100 if total > 0 else 0
    return prop_similarity, prop_mfe

def print_analysis(ref_seq, ref_structure, ref_mfe, results, prop_similarity, prop_mfe, threshold_A, threshold_B):
    """
    Print reference sequence information, prediction results for each candidate sequence, and proportions meeting threshold conditions.
    """
    print("Reference sequence:", ref_seq)
    print("Reference structure:", ref_structure)
    print("Reference MFE: {:.2f}".format(ref_mfe))
    print("=" * 40)
    # for r in results:
    #     print("Candidate sequence:", r['seq'])
    #     print("Predicted structure:", r['structure'])
    #     print("MFE: {:.2f}".format(r['mfe']))
    #     print("Structure similarity: {:.2f}%".format(r['similarity']))
    #     print("MFE difference: {:.2f} kcal/mol".format(r['mfe_diff']))
    #     print("-" * 40)
    print("Threshold A = {}: Proportion of sequences with structure similarity >= A: {:.2f}%".format(threshold_A, prop_similarity))
    print("Threshold B = {}: Proportion of sequences with MFE difference <= B: {:.2f}%".format(threshold_B, prop_mfe))

def run_rna_analysis(candidate_seqs_path, threshold_A=80.0, threshold_B=2.0, factor=2.0):
    """
      1. Read sequences from candidate sequence file
      2. Perform batch calculations (structure/MFE) for candidate and reference sequences
      3. Count proportions meeting threshold conditions and print results
    """
    candidate_seqs = get_gen_seqs(candidate_seqs_path)
    ref_seq = 'TCCAGCACTCCACGCATAACGGCGGTGGGTGGGTTGTTGTGGGAGGGGGAGGGGGAGTTATGCGTGCTACCGTGAA'
    ref_structure, ref_mfe, results = batch_analysis(candidate_seqs, ref_seq, factor)
    prop_similarity, prop_mfe = threshold_analysis(results, threshold_A, threshold_B)
    # print_analysis(ref_seq, ref_structure, ref_mfe, results, prop_similarity, prop_mfe, threshold_A, threshold_B)
    return prop_similarity, prop_mfe


### yingqing: wrap to value function

def pre_process_seqs(seqs):
    for i, line in enumerate(seqs):
        # Remove '2' if present in the line
        if '2' in line:
            line = line.replace('2', '')
        line = line.strip()[1:]  # Remove the first character as needed
        # Add fixed fragments according to requirements
        line = 'TCCAGCACTCCACGCATAAC' + line + 'GTTATGCGTGCTACCGTGAA'
        seqs[i] = line
    return seqs

def similarity_value_function(seqs, factor=5.0):
    seqs = pre_process_seqs(seqs)
    ref_seq = 'TCCAGCACTCCACGCATAACGGCGGTGGGTGGGTTGTTGTGGGAGGGGGAGGGGGAGTTATGCGTGCTACCGTGAA'
    ref_structure, ref_mfe = compute_structure(ref_seq)
    
    results = []
    for seq in seqs:
        structure, mfe = compute_structure(seq)
        similarity = compute_similarity(ref_structure, structure, factor)
        results.append(similarity)
    return results


def mfe_value_function(seqs, factor=5.0):
    seqs = pre_process_seqs(seqs)
    ref_seq = 'TCCAGCACTCCACGCATAACGGCGGTGGGTGGGTTGTTGTGGGAGGGGGAGGGGGAGTTATGCGTGCTACCGTGAA'
    ref_structure, ref_mfe = compute_structure(ref_seq)
    
    results = []
    for seq in seqs:
        structure, mfe = compute_structure(seq)
        mfe_diff = abs(mfe - ref_mfe)
        ## for maximize
        mfe_diff = -mfe_diff
        results.append(mfe_diff)
    return results


def combine_value_function(seqs, factor=5.0):
    seqs = pre_process_seqs(seqs)
    ref_seq = 'TCCAGCACTCCACGCATAACGGCGGTGGGTGGGTTGTTGTGGGAGGGGGAGGGGGAGTTATGCGTGCTACCGTGAA'
    ref_structure, ref_mfe = compute_structure(ref_seq)
    
    results = []
    for seq in seqs:
        structure, mfe = compute_structure(seq)
        similarity = compute_similarity(ref_structure, structure, factor)
        combine_score = similarity + mfe / ref_mfe
        results.append(combine_score)
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNA secondary structure analysis")
    parser.add_argument("--candidate_seqs_path", required=True,
                        help="Path to candidate RNA sequence file")
    parser.add_argument("--threshold_A", type=float, default=80.0,
                        help="Structure similarity threshold (out of 100), default 80")
    parser.add_argument("--threshold_B", type=float, default=2.0,
                        help="MFE difference threshold (kcal/mol), default 2")
    parser.add_argument("--factor", type=float, default=2.0,
                        help="Distance conversion factor, default 2")
    args = parser.parse_args()

    # Call the wrapped function for RNA analysis
    run_rna_analysis(args.candidate_seqs_path, args.threshold_A, args.threshold_B, args.factor)