# Modelling hierarchy

## Bayesian inference is not the epidemic simulator

Let `D` be access-period data and `Theta` the transmission-rate vector. EpiPvr estimates

\[
p(\Theta\mid D).
\]

This is uncertainty about biological parameters.

## Branching process

EpiPvr can then combine transmission and local field parameters in a multitype branching process to estimate an early invasion probability

\[
P_{\mathrm{est}}=1-q.
\]

That calculation concerns stochastic extinction versus establishment when infection is rare.

## Mean field

The PLOS mixture model provides deterministic population-average dynamics. Cropmix generalizes the SPT form to an arbitrary number of varieties and uses it as a consistency reference.

## Spatial Gillespie CTMC

Cropmix explicitly tracks plant states and virus-bearing vector counts at planting locations. The Gillespie algorithm generates complete stochastic field trajectories conditional on parameters.

## Two sources of uncertainty

Cropmix distinguishes:

- **parameter uncertainty** from `p(Theta | D)`;
- **process stochasticity** from the finite epidemic CTMC.

The full predictive target is

\[
p(\mathcal O\mid D,z)
=
\int p_{\mathrm{CTMC}}(\mathcal O\mid z,\Theta)
       p(\Theta\mid D)\,d\Theta.
\]

Version 0.1 exposes EpiPvr posterior draws and can accept coherent `TransmissionDraw` objects. Full Bayesian optimization across many varieties is a planned extension.
