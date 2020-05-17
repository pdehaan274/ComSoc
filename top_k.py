import copy
import numpy as np 
from collections import Counter



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

    return np.array(possible_sets[np.argmax(scores, axis=1)])


def calculate_sw(utils, winners, egal=False, nash=True):
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
        # sw = np.sum(np.log(sw))
        sw = np.prod(sw)
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

def calc_compare_score(winners, max_utility, prizes):
    res = np.zeros((max_utility.shape[0]))

    # print(f"max: {max_utility}")
    for p in winners:
        # print(f"p: {p}")
        s = p_in_max(p, max_utility)

        # print(f"s: {s}")

        prize = prizes[p]
        res += s*prize

    return np.mean(res)

if __name__ == "__main__":
    # parameters
    P = 10
    epsilon = 20
    N = 20
    k = 3
    budget = 30

    min_prize = 10
    max_prize = 25

    # initialize vectors
    base_util = make_base_util(P)
    print(f"base_util: {base_util}")

    project_prizes = make_projects(min_prize, max_prize, P)
    print(f"project_prizes: {project_prizes}")

    possible_sets = get_possible_sets(project_prizes, budget)
    # values, probs = project_probs(P)

    # print(f"there are {len(possible_sets)} possible winners")

    # best_sets = best_outcome(values, possible_sets)
    # print(f"best sets: {best_sets}")

    # create utilities for voters
    utilities = make_voter_utils(base_util, epsilon, N)

    max_utility = max_utility(possible_sets, utilities)
    # print(f"max_utility results: {max_utility}")

    for k in range(1, P):

        # calculate scores for all projects
        votes = k_approval(k, utilities, P)
        # print(f"votes: {votes}")

        # determine the winning projects
        winners = greedy_allocation(votes, project_prizes, budget)
        # print(f"winners: {winners}")

        # calculate loss
        sw = calculate_sw(utilities, winners)

        compare_score = calc_compare_score(winners, max_utility, project_prizes)
        print(f"k: {k} -- Winners: {winners}, Avg comp: {compare_score}, Avg sw: {sw}\n")

    # project_probs(P)