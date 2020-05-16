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
    return project_values, project_probs


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
        if outcome == best_outcome:
            best_sets.append(projects_set)
        elif outcome > best_outcome:
            best_sets = [project_set]
    return best_sets

##################################################################
##### functions to get the set of possible winning sets
##################################################################

def get_possible_sets(prizes, b):
    res = []
    
    get_possible_sets_helper(prizes, b, res)
    
    return remove_duplicates(res)

def get_possible_sets_helper(prizes, b, res, sub=[]):
    full = True
    for i in range(len(projects)):
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

def k_approval(k, utils):
    """
    Extract the k alternatives with the highest utility for
    all voters.
    """
    votes = np.argsort(utils, axis=1)[:, :k]

    res = Counter(votes.reshape(-1))
    for i in range(votes.shape[0]):
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



##################################################################
##### simulation functions
##################################################################

def make_base_util(P):
    """
        Create a base utility vector
    """
    
    #TODO: expand on this
    return np.random.randint(0, 10, P)

def make_projects(budget, size):
    """
    Create a dictionary with all projects and corresponding prices.
    """
    projects = {}
    #TODO: What would be a good price initialization for the projects?
    for i in range(size):
        price = 1
        # projects[i] = price
        projects[i] = np.random.randint(0, budget/2, 1)[0]
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


if __name__ == "__main__":
    # parameters
    P = 10
    epsilon = 1
    N = 20
    k = 3
    budget = 30

    # initialize vectors
    base_util = make_base_util(P)
    print(f"base_util: {base_util}")

    project_prizes = make_projects(budget, P)
    print(f"project_prizes: {project_prizes}")

    # create utilities for voters
    utilities = make_voter_utils(base_util, epsilon, N)

    for k in range(1, P):
        print(f"k: {k}")

        # calculate scores for all projects
        votes = k_approval(k, utilities)
        # print(f"votes: {votes}")

        # determine the winning projects
        winners = greedy_allocation(votes, project_prizes, budget)
        # print(f"winners: {winners}")

        # calculate loss
        sw = calculate_sw(utilities, winners)
        print(f"Avg social welfare: {sw}\n")

    project_probs(P)