<table>
<tr>
<td><img src='imgs/Subject_002.gif' width=100></td>
<td><img src='imgs/Subject_004.gif' width=100></td>
<td><img src='imgs/Subject_018.gif' width=100></td>
<td><img src='imgs/Subject_021.gif' width=100></td>
<td><img src='imgs/Subject_026.gif' width=100></td>
</tr>
</table>

[[Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SLNTKB)]

<br>

# CRISPFlow

## Accelerated Phase Contrast MRI with Use of Resolution Enhancement Generative Adversarial Neural Network  

The complex-difference reconstruction for inline super-resolution of phase-contrast flow (CRISPFlow) model was developed to accelerate phase contrast imaging. The model was built on two modified enhanced super-resolution generative adversarial neural networks (ESRGAN). The trained weights for each of the networks can be downloaded through the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SLNTKB).
<p align="center">
    <img src='imgs/inline_reconstruction.png' width=840>
    <br>
    <p align="justify">
    Velocity compensated and encoded images acquired with low resolution are reconstructed using the vendor reconstruction algorithm pipeline. The low-resolution images are sent to an external sever via a Framework for Image Reconstruction (FIRE) interface. Network 1 is used to enhance complex-difference images, which are obtained using complex-valued subtraction of velocity compensated and encoded images. Network 2 enhances velocity compensated images directly. Both networks enhance real and imaginary parts separately. The resolution-enhanced velocity compensated and encoded images, the latter obtained through complex-valued addition, are returned to the vendor pipeline to reconstruct anatomical and phase-contrast images.
    </p>
    <br><br>

</p>