# From Identity-Jacobian ILC to Iteratively Learned PA-Model Backpropagation

## Scope

This note gives a compact mathematical description of waveform iterative learning control (ILC) for
digital predistortion (DPD). It first isolates the limitation of the scalar, identity-Jacobian ILC update,
then derives an improved update that refits a forward power-amplifier (PA) model at every outer iteration
and backpropagates the measured output error through that model.

The discussion is theoretical. It contains no experimental claims. Also, “classical ILC” below means the
scalar error-injection baseline; plant-inverse, BLA-inverse, and instantaneous-gain ILC already use more PA
information and should not be conflated with the identity-Jacobian method.

## 1. Waveform DPD as an inverse problem

Let

- \(u\in\mathbb C^N\) be the finite waveform applied to the PA;
- \(F:\mathbb C^N\rightarrow\mathbb C^N\) be the calibrated PA input-output operator;
- \(d\in\mathbb C^N\) be the desired PA output;
- \(y_k=F(u_k)\) be the measured output at ILC iteration \(k\); and
- \(r_k=d-y_k\) be the output tracking error.

The implementation documentation also uses the signed residual \(e_k=y_k-d=-r_k\). This note keeps
\(r_k=d-y_k\) throughout so that the classical update has the familiar plus-error form.

The waveform-level inverse problem is

$$
\min_{u\in\mathcal U}\; \mathcal L(u),
\qquad
\mathcal L(u)=\frac12\lVert F(u)-d\rVert_2^2
             =\frac12\lVert r(u)\rVert_2^2,
$$

where \(\mathcal U\) is the feasible input set imposed by peak, RMS, PAPR, or hardware limits.

The derivation assumes a fixed input/output coordinate system. If the measured output is independently
renormalized at every iteration, the effective operator is an iteration-dependent composition of the PA and
the calibration map, rather than the physical PA map alone.

A nonlinear RF PA is generally not holomorphic as a complex map because its basis functions depend on
\(|u|\). Therefore, \(\mathbb C^N\) is treated as \(\mathbb R^{2N}\), with real inner product

$$
\langle a,b\rangle_{\mathbb R}=\operatorname{Re}(a^H b).
$$

Let \(J(u)\) denote the real-linear Fréchet derivative of \(F\):

$$
F(u+h)=F(u)+J(u)h+\mathcal O(\lVert h\rVert^2).
$$

The real adjoint \(J(u)^T\) is defined by

$$
\langle v,J(u)h\rangle_{\mathbb R}
=\langle J(u)^T v,h\rangle_{\mathbb R}.
$$

Since \(r(u)=d-F(u)\), the directional derivative of the loss is

$$
D\mathcal L(u)[h]
=-\langle r(u),J(u)h\rangle_{\mathbb R}
=\langle -J(u)^T r(u),h\rangle_{\mathbb R}.
$$

Hence the true input-space gradient is

$$
\nabla\mathcal L(u)=-J(u)^T r(u).
$$

This expression is the central distinction between output error and an input correction: the error must
be mapped back through the PA adjoint before it is a gradient in the input space.

## 2. Classical scalar ILC and its implicit approximation

The scalar linear ILC update is

$$
u_{k+1}=u_k+\mu r_k,
\qquad \mu>0.
$$

Using a first-order PA expansion at \(u_k\),

$$
\begin{aligned}
r_{k+1}
&=d-F(u_k+\mu r_k)\\
&=r_k-\mu J_k r_k+\mathcal O(\mu^2\lVert r_k\rVert^2),
\end{aligned}
$$

where \(J_k=J(u_k)\). Thus the local error dynamics are

$$
r_{k+1}\approx (I-\mu J_k)r_k.
$$

The update implicitly treats the output error as if it were already expressed in the PA input coordinates.
Equivalently, it approximates both the inverse learning operator and the gradient transport by a scaled
identity map.

For a fixed linear PA, \(F(u)=Au\), asymptotic convergence requires

$$
\rho(I-\mu A)<1,
$$

where \(\rho(\cdot)\) is the spectral radius. For a scalar complex gain \(A=g\), a real positive learning
rate must satisfy

$$
|1-\mu g|<1
\quad\Longleftrightarrow\quad
0<\mu<\frac{2\operatorname{Re}(g)}{|g|^2}.
$$

Such a \(\mu\) exists only when \(\operatorname{Re}(g)>0\). A sufficiently large PA phase rotation can
therefore turn the nominal correction into a non-contractive direction.

The same issue follows directly from the loss derivative. Along the classical step \(h=\mu r_k\),

$$
D\mathcal L(u_k)[\mu r_k]
=-\mu\langle r_k,J_k r_k\rangle_{\mathbb R}.
$$

The step is a first-order descent direction only if

$$
\langle r_k,J_k r_k\rangle_{\mathbb R}>0.
$$

That condition is not guaranteed for a general complex, nonlinear, or dynamic PA.

### 2.1 Main limitations

1. **Direction error.** Phase distortion or a negative local AM/AM slope can make
   \(\langle r_k,J_k r_k\rangle_{\mathbb R}\le 0\). The identity update can then be orthogonal to, or point
   against, the true descent direction.

2. **Scale error.** If the local PA slope is small, the output changes little although \(\mu r_k\) continues
   to accumulate at the input. If the slope is large, the same fixed \(\mu\) can overshoot. One scalar cannot
   simultaneously invert widely different singular directions.

3. **Memory and cross-sample coupling.** With PA memory, \(J_k\) is not diagonal. An error at one time index
   may require corrections at several input indices. The identity map ignores this coupling.

4. **Iteration-dependent nonlinearity.** The Jacobian changes with \(u_k\). A learning rate that is stable
   around one envelope region need not be stable after the waveform moves to another region.

5. **Saturation and infeasibility.** In deep saturation, \(J_k\) may approach zero. Classical ILC still
   injects \(\mu r_k\), even though the PA cannot produce the requested output change. If
   \(d\notin F(\mathcal U)\), then

   $$
   \min_{u\in\mathcal U}\lVert F(u)-d\rVert_2>0,
   $$

   provided that \(\mathcal U\) is compact and \(F\) is continuous. Under those assumptions,
   \(F(\mathcal U)\) is closed and no ILC law can drive the error to zero.

6. **Noise integration.** With \(y_k=F(u_k)+n_k\), the classical update contains \(-\mu n_k\) directly.
   Repeated iterations can accumulate capture noise unless filtering, averaging, regularization, or a stopping
   rule is added.

These are limitations of the identity-Jacobian baseline, not impossibility statements about every form of
ILC. A known plant inverse, an instantaneous complex gain, or a frequency-dependent learning filter can
correct some of them when its structural assumptions are valid.

## 3. An iteratively learned forward PA model

The improved method estimates the missing local PA geometry from the latest measured pair \((u_k,y_k)\).
At each outer ILC iteration, fit a forward model

$$
\widehat F_k(u)=\widehat F(u;\widehat\theta_k),
$$

with, for example, a regularized regression objective

$$
\widehat\theta_k
=\arg\min_\theta
\left\lVert \widehat F(u_k;\theta)-y_k\right\rVert_2^2
+\rho\lVert\theta\rVert_2^2.
$$

The fitted parameters \(\widehat\theta_k\) are frozen while computing iteration \(k\)'s update. Gradients
are not propagated through the regression solver, measurement process, delay estimator, or calibration
estimator. The model is refitted only after a new PA input-output pair has been observed.

This separation produces two nested loops:

1. **Identification loop:** estimate a local forward PA model from the current measurement.
2. **Learning loop:** use the frozen model's derivative to transport the measured error back to the input.

The residual remains the real measurement residual \(r_k=d-y_k\); it is not replaced by the model residual.

## 4. Real-linear backpropagation through a PA model

Let

$$
\widehat J_k=D\widehat F_k(u_k)
$$

be the frozen model derivative. Its Jacobian-vector product (JVP) maps an input perturbation to an output
perturbation,

$$
h\longmapsto \widehat J_k h,
$$

and its vector-Jacobian product (VJP) maps an output cotangent back to the input,

$$
v\longmapsto \widehat J_k^T v.
$$

### 4.1 Memory-polynomial example

For a circular memory polynomial,

$$
\widehat F_k(u)[n]
=\sum_{m=0}^{M-1}\sum_{p\in\mathcal P}
c_{p,m}\,z_m[n]
\left(\frac{|z_m[n]|}{s}\right)^{p-1},
\qquad
z_m[n]=u[(n-m)\bmod N].
$$

For one basis function

$$
\phi_p(z)=c\,z\left(\frac{|z|}{s}\right)^{p-1},
$$

the real-linear differential has both direct and conjugate terms:

$$
D\phi_p(z)[h]=a_p(z)h+b_p(z)h^*,
$$

with

$$
a_p(z)=c\frac{p+1}{2}
\left(\frac{|z|}{s}\right)^{p-1},
$$

and, for \(p>1\),

$$
b_p(z)=c\frac{p-1}{2}
\left(\frac{z}{s}\right)^2
\left(\frac{|z|}{s}\right)^{p-3}.
$$

For \(p=1\), \(b_p(z)=0\). After summing the orders at each delay into coefficient arrays \(a_m\) and
\(b_m\), the JVP is

$$
\widehat J_k h
=\sum_m\left[
a_m\odot\operatorname{roll}(h,m)
+b_m\odot\operatorname{roll}(h,m)^*
\right].
$$

The corresponding real adjoint is

$$
\widehat J_k^T v
=\sum_m\operatorname{roll}\!\left(
a_m^*\odot v+b_m\odot v^*,-m
\right).
$$

Using an ordinary complex-linear Hermitian derivative would omit the conjugate path and generally produce
the wrong gradient.

## 5. Backpropagated learning laws

### 5.1 Raw VJP update

Replacing the identity transport by the learned adjoint gives

$$
\delta_k^{\mathrm{VJP}}
=\eta\widehat J_k^T r_k,
\qquad
u_{k+1}=u_k+\delta_k^{\mathrm{VJP}}.
$$

If \(\widehat J_k=I\), this reduces exactly to scalar linear ILC with \(\eta=\mu\). The update is the
steepest-descent direction for the local model, but it is not automatically a descent direction for the true
PA when the derivative estimate is inaccurate.

With an exact derivative, \(\widehat J_k=J_k\), the first-order loss change is

$$
D\mathcal L(u_k)[\delta_k^{\mathrm{VJP}}]
=-\eta\lVert J_k^T r_k\rVert_2^2<0
$$

whenever the true gradient is nonzero.

Let the true Jacobian be

$$
J_k=\widehat J_k+E_k.
$$

The true first-order loss change along the raw VJP step is

$$
D\mathcal L(u_k)[\delta_k^{\mathrm{VJP}}]
=-\eta\left\langle J_k^T r_k,\widehat J_k^T r_k\right\rangle_{\mathbb R}.
$$

Moreover,

$$
\begin{aligned}
\left\langle J_k^T r_k,\widehat J_k^T r_k\right\rangle_{\mathbb R}
&\ge
\lVert\widehat J_k^T r_k\rVert_2
\left(
\lVert\widehat J_k^T r_k\rVert_2
-\lVert E_k\rVert_2\lVert r_k\rVert_2
\right).
\end{aligned}
$$

A sufficient local descent condition is therefore

$$
\lVert E_k\rVert_2\lVert r_k\rVert_2
<\lVert\widehat J_k^T r_k\rVert_2.
$$

This inequality makes the model-quality requirement explicit. It also shows why raw VJP can fail when the
learned gradient is small: even modest Jacobian error can dominate it.

### 5.2 Damped Gauss--Newton / Levenberg--Marquardt update

Raw gradient descent still inherits poor Jacobian scaling. A better local step minimizes the regularized
linearized residual:

$$
\delta_k
=\arg\min_\delta
\frac12\left\lVert r_k-\widehat J_k\delta\right\rVert_2^2
+\frac{\lambda_k}{2}\lVert\delta\rVert_2^2,
\qquad \lambda_k>0.
$$

Setting its derivative to zero yields the damped normal equation

$$
\boxed{
\left(\widehat J_k^T\widehat J_k+\lambda_k I\right)\delta_k
=\widehat J_k^T r_k.
}
$$

The normal operator is positive definite in the real geometry because

$$
\left\langle q,
(\widehat J_k^T\widehat J_k+\lambda_k I)q
\right\rangle_{\mathbb R}
=\lVert\widehat J_k q\rVert_2^2
+\lambda_k\lVert q\rVert_2^2>0
$$

for every nonzero \(q\). Therefore, the step can be computed by conjugate gradients using only

$$
q\mapsto \widehat J_k q
\mapsto \widehat J_k^T(\widehat J_k q)+\lambda_k q,
$$

without constructing the full \(2N\times2N\) real Jacobian.

If \(\widehat J_k=U\Sigma V^T\) is the singular-value decomposition of its real representation, then

$$
\delta_k
=V\,\operatorname{diag}\!\left(
\frac{\sigma_i}{\sigma_i^2+\lambda_k}
\right)U^T r_k.
$$

Thus well-observed directions are approximately inverted, while weak directions are suppressed instead of
being amplified without bound. The filter factor satisfies

$$
0\le \frac{\sigma}{\sigma^2+\lambda_k}
\le \frac{1}{2\sqrt{\lambda_k}}.
$$

Two limiting cases clarify the relationship to simpler ILC laws:

- If \(\widehat J_k=I\), then \(\delta_k=r_k/(1+\lambda_k)\), a damped scalar ILC step.
- If the PA is a memoryless complex gain \(g\), then

  $$
  \delta_k=\frac{g^*}{|g|^2+\lambda_k}r_k,
  $$

  which is a Tikhonov-regularized complex-gain inverse.

For a perfect linear model \(F(u)=Au\), the new residual is

$$
r_{k+1}
=\left[I-A(A^T A+\lambda I)^{-1}A^T\right]r_k.
$$

Along each nonzero singular direction, the contraction factor is

$$
\frac{\lambda}{\sigma_i^2+\lambda}.
$$

With \(\lambda=0\) and an invertible \(A\), the exact linear inverse is recovered in one idealized step.

## 6. Measurement-anchored prediction and safeguards

A forward model can have a static prediction bias at the current waveform. Using
\(\widehat F_k(u_k+\delta)\) directly would incorrectly treat that bias as a measured output change. Instead,
use the anchored prediction

$$
\widetilde y_k(\delta)
=y_k+\widehat F_k(u_k+\delta)-\widehat F_k(u_k).
$$

It satisfies \(\widetilde y_k(0)=y_k\) exactly, while retaining the model-predicted nonlinear increment.
A safeguarded candidate can be accepted only when

$$
\frac12\lVert d-\widetilde y_k(\delta)\rVert_2^2
<\frac12\lVert r_k\rVert_2^2.
$$

The update should additionally satisfy a trust region and the feasible input set:

$$
\operatorname{RMS}(\delta)
\le \tau\operatorname{RMS}(u_k),
\qquad
u_k+\delta\in\mathcal U.
$$

If the full step is rejected, test \(\beta^b\delta\), with \(0<\beta<1\), until a candidate passes the
anchored decrease and input constraints. If no candidate passes, hold \(u_k\). Model-fit failure should also
produce an explicit hold or a declared scalar-ILC fallback, rather than silently reusing an unrelated model.

## 7. What the improved method does and does not solve

The learned model changes the update from

$$
\underbrace{\delta_k=\mu r_k}_{\text{identity transport}}
$$

to either

$$
\underbrace{\delta_k=\eta\widehat J_k^T r_k}_{\text{learned gradient transport}}
$$

or

$$
\underbrace{
(\widehat J_k^T\widehat J_k+\lambda_k I)\delta_k
=\widehat J_k^T r_k
}_{\text{regularized local inverse}}.
$$

This directly addresses phase rotation, unequal local slope, and memory coupling when the forward model is
locally accurate. Iterative refitting allows the derivative to follow the waveform as it moves through the PA
operating region.

The method remains local and model-dependent:

- it has no global convergence guarantee for a nonconvex PA inverse;
- model mismatch can still rotate the learned gradient away from the true gradient;
- a poorly excited waveform cannot identify unobserved model directions;
- changing calibration across iterations changes the effective operator being learned;
- capture noise affects both the residual and the fitted model; and
- the optimized variable is one finite, repeatable waveform; arbitrary traffic requires a separate
  parameterized DPD identification stage; and
- if \(d\notin F(\mathcal U)\), no gradient or second-order approximation can make the target reachable.

In particular, at ideal hard saturation \(J_k=0\). If the learned forward model faithfully represents that
local slope, then \(\widehat J_k=0\), so both the true gradient and the right-hand side of the LM equation are
zero. Positive damping gives \(\delta_k=0\): this is a safe indication of local unresponsiveness, not a
recovery of the unreachable desired waveform.

## 8. Compact algorithm statement

At each outer iteration \(k\):

1. Apply \(u_k\), measure and align \(y_k\), and form \(r_k=d-y_k\).
2. Fit and validate \(\widehat F_k\) from the current input-output pair.
3. Freeze \(\widehat F_k\) and construct its real-linear JVP/VJP at \(u_k\).
4. Compute a raw VJP step or solve the damped normal equation by matrix-free CG.
5. Apply trust-region scaling, input projection, anchored nonlinear prediction, and backtracking.
6. Accept the first safe predicted-decrease candidate; otherwise hold or use an explicit fallback.
7. Refit only after the next physical PA measurement.

The conceptual improvement is therefore not merely “adding a model.” It replaces an implicit identity
Jacobian with an iteratively identified local differential, uses the adjoint to transport the measured error
to the PA input, and regularizes that transport before the next ILC experiment.

This construction combines established PA modeling, backpropagation, and ILC ingredients in a specific
waveform-level loop. It does not claim the first use of PA forward models, PA Jacobians, memory
backpropagation, direct learning, memory-polynomial modeling, or model-based ILC. Here “improved” means that
the update contains richer local sensitivity information and explicit numerical safeguards; it does not mean
uniform superiority over instantaneous-gain, BLA-inverse, or other plant-informed ILC methods.

For implementation-level conventions and prior-art boundaries, see
[PA Model-Backpropagation ILC Algorithm Design](algorithm_design.md) and
[ILC DPD Prior-Art and Legacy Audit](prior_art_review.md).
