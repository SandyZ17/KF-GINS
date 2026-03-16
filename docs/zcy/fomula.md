# Learning-Aided GNSS Disturbance State for Real-Time GNSS/INS ESKF

## 1. Problem setup

We keep the original GNSS/INS error-state EKF as the main estimator and only learn a low-dimensional GNSS disturbance term.

The target scenario is:

- real-time GNSS/INS integration
- urban degradation
- GNSS still available, but affected by multipath, blockage, and other unmodeled biases

The key idea is:

- do not replace the filter
- do not directly output position
- do not only tune the measurement covariance
- instead, estimate a structured low-dimensional GNSS disturbance state and inject it into the GNSS update

## 2. Nominal navigation state

The nominal state follows the KF-GINS mechanization and includes:

\[
\mathbf{x}^{n} =
\begin{bmatrix}
\mathbf{p} \\
\mathbf{v} \\
\mathbf{R}_{bn} \\
\mathbf{b}_g \\
\mathbf{b}_a \\
\mathbf{s}_g \\
\mathbf{s}_a
\end{bmatrix}
\]

where:

- \(\mathbf{p} \in \mathbb{R}^3\): position
- \(\mathbf{v} \in \mathbb{R}^3\): velocity
- \(\mathbf{R}_{bn} \in SO(3)\): attitude
- \(\mathbf{b}_g, \mathbf{b}_a \in \mathbb{R}^3\): gyroscope and accelerometer biases
- \(\mathbf{s}_g, \mathbf{s}_a \in \mathbb{R}^3\): gyroscope and accelerometer scale factor errors

## 3. Error-state model

The error-state vector is

\[
\delta \mathbf{x} =
\begin{bmatrix}
\delta \mathbf{p} \\
\delta \mathbf{v} \\
\delta \boldsymbol{\phi} \\
\delta \mathbf{b}_g \\
\delta \mathbf{b}_a \\
\delta \mathbf{s}_g \\
\delta \mathbf{s}_a
\end{bmatrix}
\in \mathbb{R}^{21}
\]

The continuous/discrete propagation remains the original KF-GINS formulation:

\[
\delta \mathbf{x}_{k+1} = \mathbf{\Phi}_k \, \delta \mathbf{x}_k + \mathbf{w}_k
\]

\[
\mathbf{P}_{k+1} = \mathbf{\Phi}_k \mathbf{P}_k \mathbf{\Phi}_k^\top + \mathbf{Q}_k
\]

This part is unchanged. The learning module only affects the GNSS update.

## 4. Original GNSS measurement update

Let the GNSS position measurement in the navigation frame be

\[
\mathbf{z}_k^{g} \in \mathbb{R}^3
\]

and the nominal predicted antenna position be

\[
\hat{\mathbf{p}}_{a,k}
\]

Then the standard position innovation in KF-GINS can be written as

\[
\mathbf{d}_k = \hat{\mathbf{p}}_{a,k} - \mathbf{z}_k^{g}
\]

or, more generally, in the linearized form

\[
\mathbf{r}_k = \mathbf{z}_k^{g} - h(\hat{\mathbf{x}}_k)
\]

with measurement Jacobian

\[
\mathbf{H}_k = \frac{\partial h}{\partial \delta \mathbf{x}}\Big|_{\hat{\mathbf{x}}_k}
\]

and covariance

\[
\mathbf{R}_k
\]

The standard EKF update is

\[
\mathbf{K}_k = \mathbf{P}_k \mathbf{H}_k^\top \left(\mathbf{H}_k \mathbf{P}_k \mathbf{H}_k^\top + \mathbf{R}_k \right)^{-1}
\]

$$
\delta \hat{\mathbf{x}}_k = \delta \hat{\mathbf{x}}_k^{-} + \mathbf{K}_k \left(\mathbf{r}_k - \mathbf{H}_k \delta \hat{\mathbf{x}}_k^{-}\right)
$$

\[
\mathbf{P}_k^{+} = (\mathbf{I}-\mathbf{K}_k \mathbf{H}_k)\mathbf{P}_k^{-}(\mathbf{I}-\mathbf{K}_k \mathbf{H}_k)^\top + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^\top
\]

## 5. Proposed GNSS disturbance state

We introduce a low-dimensional GNSS disturbance state

\[
\mathbf{d}_k^{b} \in \mathbb{R}^3
\]

defined in the navigation frame.

Its meaning is:

- a structured, time-varying GNSS measurement disturbance
- mainly induced by urban canyon effects such as multipath, partial blockage, and other unmodeled biases

This is not:

- the full state
- a covariance scaling factor
- a pseudo-position output

It is specifically a correction term for the GNSS position update.

## 6. Disturbance-corrected measurement model

We model the degraded GNSS measurement as

\[
\mathbf{z}_k^{g} = h(\mathbf{x}_k) + \mathbf{d}_k^{b} + \mathbf{n}_k
\]

where

- \(\mathbf{d}_k^{b}\) is the learned disturbance term
- \(\mathbf{n}_k \sim \mathcal{N}(0, \mathbf{R}_k)\)

Therefore the corrected innovation becomes

\[
\tilde{\mathbf{r}}_k = \mathbf{r}_k - \hat{\mathbf{d}}_k^{b}
\]

or equivalently

\[
\tilde{\mathbf{d}}_k = \mathbf{d}_k - \hat{\mathbf{d}}_k^{b}
\]

Then the Kalman update is performed using the corrected innovation:

\[
\delta \hat{\mathbf{x}}_k = \delta \hat{\mathbf{x}}_k^{-} + \mathbf{K}_k \left(\tilde{\mathbf{r}}_k - \mathbf{H}_k \delta \hat{\mathbf{x}}_k^{-}\right)
\]

This is the key difference from adaptive-noise methods:

- adaptive-noise methods modify \(\mathbf{R}_k\)
- the proposed method explicitly estimates and compensates a disturbance term

## 7. Learning model

Define a lightweight network

\[
f_{\theta}(\cdot)
\]

that outputs the estimated GNSS disturbance:

\[
\hat{\mathbf{d}}_k^{b} = f_{\theta}(\mathbf{u}_k)
\]

where \(\mathbf{u}_k\) is a real-time feature vector.

## 8. Real-time input features

To preserve real-time deployability, the network input should only use features available online.

A compact first version is:

\[
\mathbf{u}_k =
\begin{bmatrix}
\mathbf{r}_k \\
\boldsymbol{\sigma}_k^{g} \\
\mathbf{v}_k \\
\bar{\boldsymbol{\omega}}_k \\
\bar{\mathbf{a}}_k \\
\mathbf{r}_{k-1} \\
\mathbf{r}_{k-2}
\end{bmatrix}
\]

where:

- \(\mathbf{r}_k \in \mathbb{R}^3\): current GNSS innovation
- \(\boldsymbol{\sigma}_k^{g} \in \mathbb{R}^3\): GNSS standard deviations
- \(\mathbf{v}_k \in \mathbb{R}^3\): current velocity estimate
- \(\bar{\boldsymbol{\omega}}_k \in \mathbb{R}^m\): short-window IMU angular-rate statistics
- \(\bar{\mathbf{a}}_k \in \mathbb{R}^n\): short-window IMU acceleration statistics
- \(\mathbf{r}_{k-1}, \mathbf{r}_{k-2}\): recent innovation history

In the simplest implementation:

- use a small MLP
- use fixed-dimensional statistics instead of a long sequence model

## 9. Supervision target

Assume reference trajectory is available during training.

Let the ideal GNSS measurement implied by the reference state be

\[
\mathbf{z}_{k,\text{ref}}^{g}
\]

Then the target GNSS disturbance is

\[
\mathbf{d}_{k}^{b,*} = \mathbf{z}_{k}^{g} - \mathbf{z}_{k,\text{ref}}^{g}
\]

or equivalently in innovation form

\[
\mathbf{d}_{k}^{b,*} = \mathbf{r}_k - \mathbf{r}_{k,\text{ideal}}
\]

where

\[
\mathbf{r}_{k,\text{ideal}} = \mathbf{z}_{k,\text{ref}}^{g} - h(\hat{\mathbf{x}}_k)
\]

In practice, it is recommended to define the target in the same navigation-frame coordinate system used by the filter update.

## 10. Training objective

The basic loss is disturbance regression:

\[
\mathcal{L}_{\text{dist}} = \left\| \hat{\mathbf{d}}_k^{b} - \mathbf{d}_{k}^{b,*} \right\|_2^2
\]

To avoid over-correction, add regularization:

\[
\mathcal{L}_{\text{reg}} = \left\| \hat{\mathbf{d}}_k^{b} \right\|_2^2
\]

The final loss can be

\[
\mathcal{L} =
\mathcal{L}_{\text{dist}} + \lambda \mathcal{L}_{\text{reg}}
\]

Optional consistency term:

\[
\mathcal{L}_{\text{upd}} = \left\| \tilde{\mathbf{r}}_k - \mathbf{r}_{k,\text{ideal}} \right\|_2^2
\]

and

\[
\mathcal{L} =
\mathcal{L}_{\text{dist}} + \lambda_1 \mathcal{L}_{\text{reg}} + \lambda_2 \mathcal{L}_{\text{upd}}
\]

## 11. Online algorithm

At GNSS update epoch \(k\):

1. propagate nominal state and covariance using the original KF-GINS ESKF
2. compute the standard GNSS innovation \(\mathbf{r}_k\)
3. build the real-time feature vector \(\mathbf{u}_k\)
4. infer the disturbance estimate

\[
\hat{\mathbf{d}}_k^{b} = f_{\theta}(\mathbf{u}_k)
\]

1. correct the innovation

\[
\tilde{\mathbf{r}}_k = \mathbf{r}_k - \hat{\mathbf{d}}_k^{b}
\]

1. perform the standard ESKF update using \(\tilde{\mathbf{r}}_k\)

This preserves the original filter structure and only inserts one lightweight correction term before the measurement update.

## 12. Disturbance-state interpretation

The proposed \(\mathbf{d}_k^{b}\) should be interpreted as:

- a learned, low-dimensional latent representation of urban GNSS degradation
- physically anchored to the position measurement channel
- time-varying and scene-dependent

It should not be claimed as:

- the full GNSS error model
- a universal sensor bias
- a replacement for rigorous raw-observation modeling

## 13. Difference from existing common directions

### Versus adaptive covariance methods

Adaptive covariance:

\[
\mathbf{R}_k \leftarrow g_{\theta}(\cdot)
\]

Current proposal:

\[
\tilde{\mathbf{r}}_k = \mathbf{r}_k - \hat{\mathbf{d}}_k^{b}
\]

Key difference:

- covariance tuning changes confidence
- disturbance estimation changes the structured bias term itself

### Versus pseudo-observation during GNSS outage

Outage methods:

- generate substitute position or velocity updates when GNSS is absent

Current proposal:

- assumes GNSS still exists
- compensates degraded GNSS when it is biased but available

### Versus end-to-end neural navigation

End-to-end methods:

\[
\hat{\mathbf{x}}_k = f_{\theta}(\text{raw sensors})
\]

Current proposal:

- retains the full mechanization and ESKF
- learns only the hard-to-model disturbance component

## 14. Recommended concise formulation for the paper

A concise method statement:

\[
\mathbf{z}_k^{g} = h(\mathbf{x}_k) + \mathbf{d}_k^{b} + \mathbf{n}_k
\]

\[
\hat{\mathbf{d}}_k^{b} = f_{\theta}(\mathbf{u}_k)
\]

\[
\tilde{\mathbf{r}}_k = \mathbf{z}_k^{g} - h(\hat{\mathbf{x}}_k) - \hat{\mathbf{d}}_k^{b}
\]

\[
\delta \hat{\mathbf{x}}_k = \delta \hat{\mathbf{x}}_k^{-} + \mathbf{K}_k \left(\tilde{\mathbf{r}}_k - \mathbf{H}_k \delta \hat{\mathbf{x}}_k^{-}\right)
\]

This can be used as the core equation set in the method section.
