(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects e h l a d c k b g)
(:init 
(harmony)
(planet e)
(planet h)
(planet l)
(planet a)
(planet d)
(planet c)
(planet k)
(planet b)
(planet g)
(province e)
(province h)
(province l)
(province a)
(province d)
(province c)
(province k)
(province b)
(province g)
)
(:goal
(and
(craves e h)
(craves h l)
(craves l a)
(craves a d)
(craves d c)
(craves c k)
(craves k b)
(craves b g)
)))