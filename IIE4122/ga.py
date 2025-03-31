# pseudocode for Genetic Algorithm
'''
begin
Objective Function f(x), x=(x1, x2, ..., xn)T
Encode the solution into chromosomes (binary strings)
Define fitness F
Generate the initial population
Initial probabilities of crossover (pc) and mutation (pm)
while(t < Max number of generations)
    Generate new solution by crossover and mutation
    if pc > rand, Crossover; end if
    if pm > rand Mutate; end if
    Accept the new solutions if their fitness increase
    Select the current best for new generation (elitism)
end while
Decode the results and visualizations
end
'''
