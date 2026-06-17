# APRA Presentation Meaningful Images

These images were extracted from `APRA_Presentation(1).pptx` via the Pandoc media output in `APRA_Presentation_AI_media/ppt/media`.

Only high-resolution experiment figures with substantive information are kept here. Decorative 256x256 icons and lower-resolution duplicate figures from slides 7 and 10 were omitted.

| File | Source | Content note |
| --- | --- | --- |
| `slide06_main_accuracy_a3fl_curve.png` | Slide 6, `image11.png` | Main task accuracy curve under the A3FL adaptive adversarial attack, comparing APRA with FedAvg, Clip, DeepSight, FoolsGold, and RFLBAT over CIFAR-10 training rounds. |
| `slide06_main_accuracy_doba_curve.png` | Slide 6, `image12.png` | Main task accuracy curve under the DOBA persistent backdoor attack, comparing all defense methods across 700 CIFAR-10 federated learning rounds. |
| `slide06_main_accuracy_reba_curve.png` | Slide 6, `image13.png` | Main task accuracy curve under the ReBA stealthy backdoor attack, showing the stability and final accuracy of each defense method. |
| `slide06_main_accuracy_neurotoxin_curve.png` | Slide 6, `image14.png` | Main task accuracy curve under the Neurotoxin attack, comparing defense robustness when persistent poisoning targets less important parameters. |
| `slide09_backdoor_asr_a3fl_curve.png` | Slide 9, `image19.png` | Backdoor attack success rate curve under A3FL; lower values indicate stronger defense. |
| `slide09_backdoor_asr_doba_curve.png` | Slide 9, `image20.png` | Backdoor attack success rate curve under DOBA; lower values indicate stronger defense against persistent backdoor behavior. |
| `slide09_backdoor_asr_reba_curve.png` | Slide 9, `image21.png` | Backdoor attack success rate curve under ReBA, comparing how quickly each defense suppresses the attack. |
| `slide09_backdoor_asr_neurotoxin_curve.png` | Slide 9, `image22.png` | Backdoor attack success rate curve under Neurotoxin, highlighting APRA's low ASR relative to baseline defenses. |
