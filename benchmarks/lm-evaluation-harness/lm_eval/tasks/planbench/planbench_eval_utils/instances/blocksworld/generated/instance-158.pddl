(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h e d k b i g)
(:init 
(handempty)
(ontable h)
(ontable e)
(ontable d)
(ontable k)
(ontable b)
(ontable i)
(ontable g)
(clear h)
(clear e)
(clear d)
(clear k)
(clear b)
(clear i)
(clear g)
)
(:goal
(and
(on h e)
(on e d)
(on d k)
(on k b)
(on b i)
(on i g)
)))