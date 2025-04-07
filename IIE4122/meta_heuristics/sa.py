# Pseudocode for Simulated Annealing Algorithm
'''
begin
Objective function f(x)
Initialize initial temperature T0 and initial guess x0
Set final temperature T_f and max number of iterations N
Define cooling schedule T->αT, (0<α<1)
    while (T>T_f and n<N)
    Move randomly to new locations: x_n+1=x_n+randn
    Calculated δf = f_n+1(x_n+1)-f_n(x_n)
    Accept the new solution if better
        if not improved
        Generate a random number r
        Accept if p = exp[-δf/kT]>r
        end if
    Update the best x* and f*
    end while
end
'''