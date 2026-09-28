import sys
import math
import os
import subprocess
from natsort import natsorted
import pandas as pd
import numpy as np
import random

# from scripts/
# $ sudo python3 sim/PhyloJunction/sim_one.py sim/PhyloJunction/bisse_timehet.pj trs sim/PhyloJunction/ 1 10
# $ python3 sim_bisse_timehet.py bisse_timehet.pj trs ./simulate out 0 1

def parse_PJ_tree_tsv(out_path,
                      prefix,
                      tree_tsv_path,
                      tree_state_csv_paths,
                      tree_node_name,
                      idx):

    rec_tr_df = pd.read_csv(tree_tsv_path, sep='\t', header=0)

    if len (set(rec_tr_df['replicate'])) >= 2:
        exit("Detected more than one tree replicate per simulation. Exiting.")

    tr_list = rec_tr_df[tree_node_name].tolist()
    n_trs = len(tr_list)
    for i in range(n_trs):
        j = i + 1

        # write parsed tree file
        parsed_tr_path = prefix + "." + str(idx + i) + ".tre"
        with open(out_path + parsed_tr_path, "w") as outfile:
            print(tr_list[i], file=outfile)

        # write states to csv file
        parsed_states_path = prefix + "." + str(idx + i) + ".dat.csv"
        with open(out_path + parsed_states_path, "w") as outfile:
            with open(tree_state_csv_paths[i]) as infile:
                lines = infile.readlines()
                lines[-1] = lines[-1].rstrip() # remove the last new line character
                content = "taxa,data\n" + "".join(l.replace("\t", ",") for l in lines) # add header
                print(content, file=outfile)


def parse_PJ_scalar_tsv(out_path, prefix, scalar_csv_path, idx):

    # write scalar csv file
    with open(scalar_csv_path, "r") as infile:
        header = infile.readline().split(",")
        header = [ 'log10_' + x for x in header[:-1] ] + [header[-1]]
        rv_names = ",".join(header[2:]) # skip sim and repl

        for i, line in enumerate(infile):
            print("i: " + str(i) + " line " + line)
            vals = [ math.log(float(x), 10) for x in line.split(',')[2:-1] ] + [line.split(',')[-1]]
            vals = ','.join([str(x) for x in vals]).rstrip()

            with open(out_path + prefix + "." + str(idx + i) + ".labels.csv", "w") as outfile:
                print(rv_names + vals, file=outfile)


if __name__ == "__main__":

    os.nice(19)
    ###########################
    # get and parse arguments #
    ###########################
   
    pj_script_path = sys.argv[1]
    if not os.path.isfile(pj_script_path):
        exit("Could not find " + pj_script_path + ". Exiting.")

    # e.g. trs_1
    tree_node_name = sys.argv[2]

    pj_out_path = sys.argv[3]
    if not pj_out_path.endswith("/"):
        pj_out_path += "/"

    prefix = sys.argv[4]
    idx = int(sys.argv[5]) # used to name files

    n_batches = sys.argv[6] # PJ's number of samples

    ##################
    # preparing dirs #
    ##################

    # output of this script
    out_path = pj_out_path # + "parsed_sim_output/"
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
        print("Created", out_path)

    #####################
    # editing PJ script #
    #####################

    head, tail = os.path.split(pj_script_path)
    if head == '':
        head = '.'
    if not head.endswith("/"):
        head += "/"
    random.seed(idx)
        
    for i in range(int(n_batches)):
        
        tree_found = False
        while True:
            if tree_found == True:
                break

            similar_rates = True
            while (similar_rates):
                birth = np.random.uniform(low=0.1, high=1, size=2)
                turnover = np.random.uniform(low=0, high=0.9, size=2)
                trans_01 = np.array([(birth[0]+birth[1])/20])#np.random.uniform(low=0.01, high=0.1, size=1)
                trans_10 = trans_01#np.random.uniform(low=0.01, high=0.1, size=1)
                death = birth*turnover
                birth_ratio = birth[0]/birth[1]
                death_ratio = death[0]/death[1]
                print("births", birth)
                print("deaths", death)
                print("transitions", trans_01)

                if birth_ratio < 1:
                    birth_ratio = birth[1]/birth[0]
                if death_ratio < 1:
                    birth_ratio = birth[1]/birth[0]
                if birth_ratio < 1.1 and death_ratio < 1.1:
                    similar_rates = True
                    print("rates are too similar!")
                else:
                    similar_rates = False
            # smallest_turnover = min(turnover)
            # print("smallest turnover is", smallest_turnover)
            # if (smallest_turnover > 0.3):
            #     age_upper = 20
            # elif (smallest_turnover > 0.2):
            #     age_upper = 30
            # elif(smallest_turnover > 0.1):
            #     age_upper = 50
            max_ages = [np.random.lognormal(3.249, 0.68987)]
        
            tree_found = False
            for max_age in max_ages:
                det_rates_string = ""
                csv_header = "sample,replicate,birth_rate_t1,birth_rate_t2,death_rate_t1,death_rate_t2,trans_01, trans_10"
                print("max age:", max_age)
                if tree_found == True:
                    break
                csv_vals = "1,1," + str(birth[0]) +"," + str(birth[1]) + "," + str(death[0]) +"," + str(death[1]) + "," + str(trans_01[0]) + "," + str(trans_10[0])

                #print("n taxa is " + str(n_taxa) + " max age is " + str(max_age))

                det_rates_string =  "det_birth_rate_s1" + ':= sse_rate(name="birth_rate_s1' + '", value='  + str(birth[0]) + ', event="speciation", states=[0,0,0]'  +")\n"
                det_rates_string =  det_rates_string + "det_birth_rate_s2" + ':= sse_rate(name="birth_rate_s2' + '", value='  + str(birth[1]) + ', event="speciation", states=[1,1,1]'  +")\n"
                det_rates_string = det_rates_string + "det_death_rate_s1" + ':= sse_rate(name="death_rate_s1' + '", value='  + str(death[0]) + ', event="extinction", states=[0]'  +")\n"
                det_rates_string = det_rates_string + "det_death_rate_s2" + ':= sse_rate(name="death_rate_s2' + '", value='  + str(death[1]) + ', event="extinction", states=[1]'  +")\n"
                det_rates_string = det_rates_string + "det_trans_rate_01" + ':= sse_rate(name="q01' + '", value='  + str(trans_01[0]) + ', states=[0,1], event="transition"'  +")\n"
                det_rates_string = det_rates_string + "det_trans_rate_10" + ':= sse_rate(name="q10' + '", value='  + str(trans_10[0]) + ', states=[1,0], event="transition"'  +")\n"

                count = i + idx
                sim_prefix = prefix + '.' + str(count)
                parsed_pj_script_path = head + tail.replace(".pj", "_" + str(count) + "_parsed.pj")
                with open(parsed_pj_script_path, "w") as outfile:
                    with open(pj_script_path, "r") as infile:
                        for line in infile:
                            line = line.rstrip()
                            # print("line: ", line)
                            # line = line.replace("replace_birth", str(birth[0]))
                            # line = line.replace("replace_death", str(death[0]))
                            #line = line.replace("replace_n_taxa", str(n_taxa))
                            line = line.replace("replace_max_age", str(max_age))
                            line = line.replace("replace_det_rates", str(det_rates_string))
                            if line.startswith("n_batches <-"):
                                print("n_batches <- " + n_batches, file=outfile)
                            #elif line.startswith("trs ~"):
                            #    line = line.replace("trs ~", tree_node_name + " ~")
                            #    print(line, file=outfile)
                            else:
                                print(line, file=outfile)

                print("Successfully updated PhyloJunction script (in", parsed_pj_script_path + ")")
                        
                ###########
                # call PJ #
                ###########

                call_pj = True
                if call_pj:
                    pj_args = ["nice", "-n", "19", "pjcli", parsed_pj_script_path, "-d", "-r", str(count), "-o", pj_out_path, "-p", sim_prefix ]

                    p = subprocess.Popen(pj_args, stdout=subprocess.PIPE)
                    pj_bytes = p.communicate()[0]
                    if p.returncode != 0:                     
                        print("PhyloJunction error (maybe time out)")
                        continue
                    pj_msg = pj_bytes.decode('utf-8')
                    print("PhyloJunction log:\n", pj_msg)



                #####################
                # reading PJ output #
                #####################

        
                # csv_header = csv_header+ ",model_type"
                # csv_vals = csv_vals + ",1"
                with open(pj_out_path + prefix + "." + str(count) + "_scalar_rvs_repl1.csv", "w") as file:
                    file.write(csv_header + "\n")
                    file.write(csv_vals)
                print(pj_out_path + prefix + "." + str(count) + "_scalar_rvs_repl1.csv")

                # get tree tsv path
                tree_tsv_path = pj_out_path + sim_prefix + "_" + tree_node_name + "_reconstructed.tsv"
                if not os.path.isfile(tree_tsv_path):
                    exit("Could not find " + tree_tsv_path + ". Exiting.")

                # get tree state csv paths
                tree_state_csv_paths = natsorted([pj_out_path + f for f in os.listdir(pj_out_path) \
                                        if f.endswith("repl1.tsv") and f.startswith(sim_prefix + "_" + tree_node_name + "_")])
                if len(tree_state_csv_paths) == 0:
                    exit("Could not find any .tsv file containing tip states. Exiting.")

                # # get scalar csv path
                scalar_csv_path = pj_out_path + sim_prefix + "_scalar_rvs_repl1.csv"
                if not os.path.isfile(scalar_csv_path):
                    exit("Could not find " + scalar_csv_path + ". Exiting.")
                
                stats = pd.read_csv(pj_out_path + sim_prefix + "_trs_stats.csv")
                print("size: " + str(stats["n_extant"].iloc[0]))
                if int(stats["n_extant"].iloc[0]) >= 100 and int(stats["n_extant"].iloc[0]) <= 1000:
                    print("taxa size okay!")
                    tree_found = True



                #####################
                # parsing PJ output #
                #####################

                    parse_PJ_tree_tsv(out_path,
                                    prefix,
                                    tree_tsv_path,
                                    tree_state_csv_paths,
                                    tree_node_name,
                                    count)

                    parse_PJ_scalar_tsv(out_path,
                                            prefix,
                                            scalar_csv_path,
                                            count)
                
                files = [pj_out_path + sim_prefix + "_" + tree_node_name + "_sample1_repl1_anc_states.tsv",
                                 pj_out_path + sim_prefix + "_" + tree_node_name + "_sample1_repl1.tsv",
                                 pj_out_path + sim_prefix + "_" + tree_node_name + "_annotated_complete.tsv",
                                 pj_out_path + sim_prefix + "_" + tree_node_name + "_reconstructed.tsv",
                                pj_out_path + sim_prefix + "_" + tree_node_name + "_reconstructed.tsv",
                                 pj_out_path + sim_prefix + "_scalar_rvs_repl1.csv"]
                for f in files:
                    if os.path.isfile(f):
                        os.remove(f)
                    break


                ###########
                # cleanup #
                ###########
                
        os.remove(parsed_pj_script_path)

