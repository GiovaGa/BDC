import numpy as np
import sys as sus

def print_point(p, group):

    for x in p:
        print("{0:.2f}".format(x) + ",", end="")
    print(group)

def rotate_point(p, alpha):

    rotation_matrix = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha),  np.cos(alpha)]
    ])

    return rotation_matrix @ p 

def main():

    #Assumptions: K>=2, K <= N

    if len(sus.argv) != 3:
        print("Usage: G45GEN.py <N> <K>")
        sus.exit(1)

    N, K = int(sus.argv[1]), int(sus.argv[2])

    #The only B point is the origin
    print_point([0, 0], "B")

    #K-1 A points are launched in the stratosphere
    #This ensures that they will occupy a center in both clusterings
    for alpha in np.linspace(start=0, stop=2*np.pi*(K-2)/(K-1), num=K-1):
        print_point(rotate_point([N**2,0],alpha), "A")

    for i in range(N-K):
        print_point([1, 0], "A")
 
main()