import numpy as np 
from collections import Counter


def k_approval(k, utils):
    """
    Extract the k alternatives with the highest utility for
    all voters.
    """
    votes = np.argsort(utils, axis=1)[:, :k]
    return Counter(votes.reshape(-1))


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

def make_voter_util(base_util, epsilon, n):
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

def calculate_sw(utils, winners):
    """
    Calculate the utilitarian social welfare, averaged over
    the amount of voters.
    """
    sw = utils[:, list(winners)]
    sw = np.sum(sw, axis=1)
    sw = np.mean(sw)
    return sw




if __name__ == "__main__":
    # parameters
    P = 10
    epsilon = 1
    N = 500
    k = 3
    budget = 30

    # initialize vectors
    base_util = make_base_util(P)
    print(f"base_util: {base_util}")

    project_prizes = make_projects(budget, P)
    print(f"project_prizes: {project_prizes}")

    # create utilities for voters
    utilities = make_voter_util(base_util, epsilon, N)

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
