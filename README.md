# Candidate for a FaustAI to explore the Unknown

What is this about? I want to recite Faust, Goethe for the answer:
> Der Erdenkreis ist mir genug bekannt.  
> Nach drüben ist die Aussicht uns verrannt;  
> Tor, wer dorthin die Augen blinzelnd richtet,  
> Sich über Wolken seinesgleichen dichtet!

This tension illustrates that a complete description of the tissue (our "Erdenkreis" as per Lucy) may not be possible with our current methods, and therefore, we may resort to a hyperspace approach of the neural network, since its understanding of the system could differ from ours, which, however, could prove beneficial to us.

## Briefer about PINNs

[PINNs](https://maziarraissi.github.io/PINNs/) can be designed to solve two classes of problems:
- data-driven solution (forward problem)
- data-driven discovery (inverse problem)  

of differential equations e.g. partical differential equations (PDE).  

Here we implemented the **data-driven discovery** given noisy and incomplete measurements.  
It is important to understand that the PDEs (that govern a given data-set), or in generell the xDEs, get embeded into the learning process of the NN.  
Explicitly speaking, the PDEs get embeded into the cost function of the NN. This is done using the DeepXDE package.  
With that, the embeded PDEs act as a regularization agent that limits the space of admissible solutions of the NN training.  
The PINN alone does not find any unknown/missing terms of the PDE problem.  
**It only adjusts the unknown PDE parameters** as part of its cost function.
