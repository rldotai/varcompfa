"""
Implementation of common solutions for MDPs/RL in the case where the problem
can be directly analyzed in terms of matrices. 
"""

import numpy as np 
import mdpy


def mc_return(P, r, Γ):
    assert(mdpy.is_stochastic(P))
    I = np.eye(len(P))
    return np.linalg.pinv(I - P @ Γ) @ r

def ls_weights(P, r, Γ, X):
    assert(mdpy.is_stochastic(P))
    assert(X.ndim == 2)
    assert(len(X) == len(P))
    value = mc_return(P, r, Γ)
    dist  = mdpy.stationary(P)
    D     = np.diag(dist)
    return np.linalg.pinv(X.T @ D @ X) @ X.T @ D @ value

def ls_values(P, r, Γ, X):
    weights = ls_weights(P, r, Γ, X)
    return X @ weights

def td_weights(P, r, Γ, Λ, X):
    assert(mdpy.is_stochastic(P))
    assert(X.ndim == 2)
    assert(len(X) == len(P))
    assert(mdpy.is_diagonal(Γ))
    assert(mdpy.is_diagonal(Λ))
    r_lm = (I - P @ Γ @ Λ) @ r
    P_lm = I - pinv(I - P @ Γ @ Λ) @ (I - P @ Γ)
    A = X.T @ D @ (I - P_lm) @ X
    b = X.T @ D @ r_lm
    return np.linalg.pinv(A) @ b

def td_values(P, r, Γ, Λ, X):
    return X @ td_weights(P, r, Γ, Λ, X)
    
def lambda_return(P, r, Γ, Λ, v_hat):
    # Incorporate next-state's value into expected reward
    r_hat = r + P @ Γ @ (I - Λ) @ v_hat
    # Solve the Bellman equation
    return np.linalg.pinv(I - P @ Γ @ Λ) @ r_hat

def sobel_variance(P, R, Γ):
    assert(mdpy.is_stochastic(P))
    assert(P.shape == R.shape)
    assert(mdpy.is_diagonal(Γ))
    ns = len(P)
    r = (P * R) @ np.ones(ns)
    v_pi = mc_return(P, r, Γ)
    
    # Set up Bellman equation
    q = -v_pi**2
    for i in range(ns):
        for j in range(ns):
            q[i] += P[i,j]*(R[i,j] + Γ[j,j]*v_pi[j])**2
    # Solve Bellman equation
    return np.linalg.pinv(I - P @ Γ @ Γ) @ q

def second_moment(P, R, Γ, Λ):
    assert(mdpy.is_stochastic(P))
    assert(P.shape == R.shape)
    assert(mdpy.is_diagonal(Γ))
    assert(mdpy.is_diagonal(Λ))
    ns = len(P)
    # Here the MC-return is both the lambda return and its approximation
    v_lm = mc_return(P, r, Γ)
    γ = np.diag(Γ)
    λ = np.diag(Λ)
    
    # Compute reward-like transition matrix
    R_bar = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            R_bar[i,j] = R[i,j]**2 \
                + (γ[j] * (1-λ[j])*v_lm[j])**2 \
                + 2*( γ[j] * (1 - λ[j]) * R[i,j] * v_lm[j] ) \
                + 2*( γ[j] * λ[j] * R[i,j] * v_lm[j]) \
                + 2*( (γ[j]**2)*λ[j]*(1-λ[j]) * (v_lm[j]**2) )
    # Set up Bellman equation for second moment
    r_bar = (P * R_bar) @ np.ones(ns)
    
    # Solve the Bellman equation
    return np.linalg.pinv(I - P @ Γ @ Γ @ Λ @ Λ) @ r_bar

def lambda_second_moment(P, R, Γ, Λ, v_hat):
    assert(mdpy.is_stochastic(P))
    assert(P.shape == R.shape)
    assert(mdpy.is_diagonal(Γ))
    assert(mdpy.is_diagonal(Λ))
    ns = len(P)
    # Expected immediate reward
    r = (P * R) @ np.ones(ns)
    # Lambda return may be different from approximate lambda return
    v_lm = lambda_return(P, r, Γ, Λ, v_hat)
    
    # Get per-state discount and bootstrapping
    γ = np.diag(Γ)
    λ = np.diag(Λ)
    
    # Compute reward-like transition matrix
    R_bar = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            R_bar[i,j] = R[i,j]**2 \
                + (γ[j] * (1-λ[j])*v_lm[j])**2 \
                + 2*( γ[j] * (1 - λ[j]) * R[i,j] * v_hat[j] ) \
                + 2*( γ[j] * λ[j] * R[i,j] * v_lm[j]) \
                + 2*( (γ[j]**2)*λ[j]*(1-λ[j]) * (v_hat[j]*v_lm[j]) )
    # Set up Bellman equation for second moment
    r_bar = (P * R_bar) @ np.ones(ns)
    
    # Solve the Bellman equation
    return pinv(I - P @ Γ @ Γ @ Λ @ Λ) @ r_bar