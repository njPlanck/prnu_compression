
# How does Compression degrade PRNU-based Source Camera Identification?
With the continued growth and adoption of digital media, the need to attribute digital images to
their sources after they have been acquired remains a fundamental problem in multimedia forensics.
So far, many studies have identified that Source Camera Identification (SCI) using 
Photo Response Non-Uniformity (PRNU) degrades under compression. These studies mostly assess baseline 
JPEG and its impact at different Quality Factors (QFs). However, for a given QF, different images 
compress to different sizes. This is problematic for evaluating the effect of compression in the real-world setting,
as in most networks and storage systems the target size is often the fixed parameter while the QF varies per image. 
In this paper we  evaluate the SCI performace under compression at different target bitrates,
simulating real-world strict bandwidth and storage constraints. 
We also extend this evaluation to the more recent compression schemes like JPEG 2000, JPEG XR and JPEG XL.
This is to provides a codec recommendation for severe image degradation at very low bitrates that mostly 
preserve the integrity of the forensic signal required for SCI. For the experiment, the Adobe Fivek dataset 
was used with 17 cameras selected between three different camera brands (NIKON, Canon and Sony). 