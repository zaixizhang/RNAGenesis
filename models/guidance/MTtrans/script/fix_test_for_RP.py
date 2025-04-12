
##  generating the hand craft featuress


##

test_set_tid = []
trainval_dict = {}
test_set_dict = {}
for cell_line in ['muscle',"PC3","293T"]:

    seed = 41

    df = processed_dict[cell_line]
    df_to_sample = df.query("`T_id` not in @test_set_tid") 
    df_overlap = df.query("(`T_id` in @test_set_tid)")

    # the number to sample reduce as the list cumulated from previous cell type
    # what to sample 
    n_2_sample = int(0.1*df.shape[0]) - df_overlap.shape[0]
    sampled_subset = df_to_sample.sample(n=n_2_sample,random_state=seed)

    # merge newly sampled with those in the list
    test_set_df = sampled_subset.append(df_overlap)
    trainval_dict[cell_line] = pd.concat([df,test_set_df]).drop_duplicates(keep=False)
    test_set_dict[cell_line] = test_set_df
    # adding new Tids
    test_set_tid += sampled_subset.T_id.values.tolist() 

for cell_line in ["muscle","PC3","293T"]:
    trainval_dict[cell_line].to_csv(pj(f"RP_{cell_line}_train_val.csv"))
    test_set_dict[cell_line].to_csv(pj(f"RP_{cell_line}_test.csv"))