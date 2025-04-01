# Pseudo Particle Swarm Optimization
'''
begin
    Objective function f(x), x=(x1, x2, ..., xp)T
    Initialize locations x_i and velocity v_i of n particles
    Initial minimum f_min^{t=0} = min{f(x1), ..., f(xn)} (at t=0)
    while(criterion)
    t=t+1 (pseudo time or iteration counter)
        for loop over all n particles and all p dimensions
        Generate new velocity v_i^{t+1} using equation (5.1)
        Calculate new locations x_i^{t+1} = x_i^{t} + v_i^{t+1}
        Evaluate objective functions at new locations x_i^{t+1}
        Find the current minimum f_min^{t+1}
        end for
        Find the current best x_i* and current global best g*
    end while
    Output the results x_i* and g*
end
'''
