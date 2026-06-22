# Pythonic process based modeling
Python reimplementation of [ProBMoT](http://probmot.ijs.si).  

## Key ideas
1. Simple and intuitive usage
2. Fast(er) computations
3. Scalability

### Simplicity 
Probmot is *hard to use*. User needs to write at least two seperate code files - a java-like `.pbm` file for the formalization and an `.xml` file for the task specification. User expirience is overly complicated. 

In `PyBM`, we aim to provide a simple arhitecture that is **easy to understand and reproduce**. 

### Speed 


## Process based modeilng as a type theory
Definitions of variables, constants and entities are trivial. 
$$
\frac{x : \text{str}}{ \operatorname{Var}(x) : \operatorname{VAR}}
\quad
\frac{y : \text{str}}{ \operatorname{Const}(y) : \operatorname{CONST}}
$$
$$
\frac{v_1, \ldots , v_n : \operatorname{VAR}, 
c_1, \ldots , c_m : \operatorname{CONST}}
{\operatorname{Entity}(v_1, \ldots v_n, c_1, \ldots, c_m) : 
\operatorname{ENTITY}}
$$
We can recursivly define a process
$$
\frac{e_1, \ldots, e_n : \operatorname{ENTITY}}
{\operatorname{Process}(e_1, \ldots, e_n) : \operatorname{PROCESS}}
\quad
\frac{e_1, \ldots, e_n : \operatorname{ENTITY}
p_1, \ldots, p_m : \operatorname{PROCESS}
}
{\operatorname{Process}(e_1, \ldots, e_n, p_1, \ldots, p_m) : \operatorname{PROCESS}}
$$
and so one. Using this recursive definitions, we can ease up the formalisaton. We can then define Induction of the models easily. 