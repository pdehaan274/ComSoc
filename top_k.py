import copy
import numpy as np 
from collections import Counter
from tqdm import tqdm
from sklearn.preprocessing import normalize

import argparse

#################################################################
#### functions to use epistemic accuracy                     ####
#################################################################

def project_probs(P):
    """
    Set objective "true" values for each project. From these values
    generate the probability with which a voter should vote for
    a project.
    """
    values = np.random.normal(1, 0.5, P)
    values[values<0] = 0
    project_probs = values / np.sum(values)
    return values, project_probs


def best_outcome(values, possible_sets):
    """
    Calculate the best outcomes given that all projects have a true
    utility value.
    """
    max_outcome = 0
    best_sets = []
    for project_set in possible_sets:
        vals = values[project_set]
        outcome = np.sum(vals)
        if outcome == max_outcome:
            best_sets.append(project_set)
        elif outcome > max_outcome:
            best_sets = [project_set]
    return best_sets

def epistemic_topk(probs, k, N):
    """
    Generate votes given the epistemic probability of a voter
    voting for a project.
    """
    votes = []
    projects = list(range(len(probs)))
    # Create noisy prob vectors for voters
    probs = np.tile(probs, (N,1))
    noise = np.random.normal(0.07, 0.05, (N, len(projects)))
    probs += noise
    # Set all negative probs to 0 and 
    probs[probs < 0] = 0
    probs = normalize(probs, "l1", axis=1)
    # Every voters uses prob distr to vote
    for i in range(N):
        vote = np.random.choice(projects, k, replace=False, p=probs[i])
        votes += list(vote)
    return Counter(votes)

##################################################################
##### functions to get the set of possible winning sets
##################################################################

def get_possible_sets(prizes, b):
    res = []
    
    get_possible_sets_helper(prizes, b, res)
    
    res = remove_duplicates(res)

    return np.array(res)

def get_possible_sets_helper(prizes, b, res, sub=[]):
    full = True
    for i in range(len(prizes)):
        if i in sub:
            continue
        
        prize = prizes[i]
        if prize <= b:
            sub2 = copy.copy(sub)
            sub2.append(i)
            get_possible_sets_helper(prizes, b-prize, res, sub2)
            full = False
            
    if full:
        res.append(sub)

def find_duplicates(res, a):
    idx = []
    for i in range(len(res)):
        a_ = res[i]
        if a == a_:
            continue
        
        dupe = True
        for p in a_:
            if p not in a:
                dupe = False
        if dupe:
            idx.append(i)
            
    return idx
        
        
def remove_duplicates(res):
    duplicates = []
    new_res = []
    for i in range(len(res)):
        if i in duplicates:
            continue
        
        a = res[i]
        new_res.append(a)
        
        dupes = find_duplicates(res, a)
        
        duplicates += dupes
        
    return new_res

##################################################################
##### voting rule functions
##################################################################

def k_approval(k, utils, P):
    """
    Extract the k alternatives with the highest utility for
    all voters.
    """
    votes = np.argsort(utils, axis=1)[:, :k]
    res = Counter(votes.reshape(-1))
    for i in range(P):
        if i not in res:
            res[i] = 0

    return res


def greedy_allocation(votes, projects, budget):
    """
    Use greedy approval to allocate the budget to the most-preferred
    projects of the voters. 
    """
    # Create top-k ranking/approval
    allocation = set()
    most_votes = votes.most_common()

    lowest_price = np.min(list(projects.values()))
    for project in most_votes:
            
        project_to_add = project[0]
        price = projects[project_to_add]

        if budget >= price:
            allocation.add(project_to_add)
            budget -= price

        if budget < lowest_price:
            break

    return allocation


##################################################################
##### simulation functions
##################################################################

def make_base_util(P):
    """
        Create a base utility vector
    """
    
    #TODO: expand on this
    return np.random.randint(0, 10, P)

def make_projects(min_prize, max_prize, size):
    """
    Create a dictionary with all projects and corresponding prices.
    """
    projects = {}
    #TODO: What would be a good price initialization for the projects?
    for i in range(size):
        # price = 1
        # projects[i] = price
        projects[i] = np.random.randint(min_prize, max_prize, 1)[0]
    return projects

def make_voter_utils(base_util, epsilon, n):
    """
    Create utility profiles for n voters. All profiles are based on a base
    utility profile adjusted with random noise. Final voting profiles are
    all normalized with values between 0 and 1.
    """
    base = np.array([base_util for _ in range(n)])

    noise = np.random.normal(0,epsilon,base.shape)
    utils = base_util + noise

    u_max = np.max(utils, axis=1)
    u_min = np.min(utils, axis=1)
    
    ampl = u_max - u_min

    utils = utils - u_min.reshape(-1, 1)

    utils /= ampl.reshape(-1, 1)

    utils /= np.sum(utils, axis=1).reshape(-1,1)
    return utils

##################################################################
##### Calculate utilities
##################################################################

def max_utility(possible_sets, utilities):
    """
        Calculates the best possible result for every voter based on their utilities
    """
    scores = np.zeros((utilities.shape[0], possible_sets.shape[0]))

    for i in range(len(possible_sets)):
        sub_utils = utilities[:, possible_sets[i]]
        scores[:, i] = np.sum(sub_utils, axis=1)

    best_personal_outcomes = np.array(possible_sets[np.argmax(scores, axis=1)])
    best_total_utility = np.max(np.sum(scores, axis=0))

    return best_personal_outcomes, best_total_utility


def calculate_sw(utils, winners, egal=False, nash=False):
    """
    Calculate the utilitarian social welfare, averaged over
    the amount of voters.
    """
    N = utils.shape[0]
    sw = utils[:, list(winners)]
    sw = np.sum(sw, axis=1)
    if egal:
        sw = np.min(sw)
    #TODO: log of nie
    elif nash:
        sw = np.sum(np.log(sw))
        # sw = np.prod(sw)
        sw *= (1/N)
    else:
        sw = np.mean(sw)
    return sw
 
def p_in_max(p, max_utility):
    """
        Determines if a project is in the best results for the voters
    """
    res = np.zeros((max_utility.shape[0]))
    for i in range(max_utility.shape[0]):
        if p in max_utility[i]:
            res[i] = 1

    return res

def calc_compare_score(winners, max_set, prizes):
    """
    Calculate what part of the budget is allocated to voters' favorite
    projects.
    """
    res = np.zeros((max_set.shape[0]))

    for p in winners:
        s = p_in_max(p, max_set)

        prize = prizes[p]
        res += s*prize

    return np.mean(res)

def calc_dist_score(winners, max_utility, utilities):
    """
    Calculate the distortion.
    """
    res = np.zeros((max_utility.shape[0]))

    for i in range(utilities.shape[0]):

        current_score = np.sum(utilities[i, list(winners)])

        if current_score == 0:
            res[i] = 1000
            continue
        
        max_score = np.sum(utilities[i, max_utility[i]])

        res[i] = max_score/current_score

    return np.mean(res)

def main(wf, P = 10, epsilon = 20, N = 20, budget = 30, 
                        min_prize = 10, max_prize = 25):


    wf.write(f"{P},{epsilon},{N},{budget},{min_prize},{max_prize},")

    # create the prizes for each project and determine all possible winners
    project_prizes = make_projects(min_prize, max_prize, P)
    # project_prizes= {0:10, 1:20, 2:3, 3:8, 4:5, 5:5, 6:10, 7:10, 8:30, 9:25}
    possible_sets = get_possible_sets(project_prizes, budget)

    for i in range(P):
        wf.write(f"{project_prizes[i]},")

    # create utilities for voters
    base_util = make_base_util(P)
    # base_util = np.asarray([5.8, 2.3, 29.3, 9.5, 14.8, 9.4, 5.4, 14.9, 6.7, 1.9])
    utilities = make_voter_utils(base_util, epsilon, N)

    for i in range(P-1):
        wf.write(f"{base_util[i]},")

    wf.write(f"{base_util[P-1]}")
    # determine the best winning set for all voters
    max_set, max_total_utility = max_utility(possible_sets, utilities)

    for k in range(1, P):
        # calculate scores for all projects
        votes = k_approval(k, utilities, P)
        
        # determine the winning projects
        winners = greedy_allocation(votes, project_prizes, budget)

        # calculate loss
        util_score = calculate_sw(utilities, winners)
        egal_score = calculate_sw(utilities, winners, egal=True)
        # nash_score = calculate_sw(utilities, winners, nash=True)
        nash_score = 0

        comp_score = calc_compare_score(winners, max_set, project_prizes)
        dist_score = max_total_utility / util_score
        
        winners = ''.join(str(num) for num in list(winners))

        wf.write(f",{str(winners)},{util_score},{egal_score},{nash_score},{comp_score},{dist_score}")

    wf.write("\n")
    # project_probs(P)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--loops', type=int, default=1000,
                    help='an integer for the accumulator')

    parser.add_argument('--P', type=int, default=10,
                        help='an integer for the accumulator')
    parser.add_argument('--epsilon', type=int, default=20,
                    help='an integer for the accumulator')
    parser.add_argument('--N', type=int, default=1000,
                    help='an integer for the accumulator')
    parser.add_argument('--budget', type=int, default=30,
                        help='an integer for the accumulator')
    parser.add_argument('--min', type=int, default=10,
                        help='an integer for the accumulator')
    parser.add_argument('--max', type=int, default=25,
                        help='an integer for the accumulator')

    args = parser.parse_args()

    file_name = f"results_P{args.P}_ep{args.epsilon}_N{args.N}"
    file_name += f"_B{args.budget}_min{args.min}_max{args.max}_L{args.loops}.csv"
    
    print(f"file name: {file_name}")
    wf = open(f"results/{file_name}", "w")

    wf.write(f"P,epsilon,N,budget,min_prize,max_prize,")

    for i in range(args.P):
        wf.write(f"p_{i},")

    for i in range(args.P):
        wf.write(f"u_{i},")

    for i in range(1, args.P-1):
        wf.write(f"k_{i}_winners,")
        wf.write(f"k_{i}_util,")
        wf.write(f"k_{i}_egal,")
        wf.write(f"k_{i}_nash,")
        wf.write(f"k_{i}_comp,")    
        wf.write(f"k_{i}_dist,")    

    wf.write(f"k_{args.P-1}_winners,")
    wf.write(f"k_{args.P-1}_util,")
    wf.write(f"k_{args.P-1}_egal,")
    wf.write(f"k_{args.P-1}_nash,")
    wf.write(f"k_{args.P-1}_comp,")
    wf.write(f"k_{args.P-1}_dist\n")   

    for _ in tqdm(range(args.loops)):
        main(wf, P = args.P, epsilon = args.epsilon, N = args.N, 
             budget = args.budget, min_prize = args.min, max_prize = args.max)

    wf.close()