# Real model update

The previous four-class model had 86.97% test accuracy but only 32.54% balanced accuracy. The headline accuracy was dominated by the Normal class, while Fusion and Supraventricular recall were near zero.

The new version makes **Normal vs Abnormal** the primary screening model and uses a secondary three-class model for subtype information:

- Normal
- Ventricular
- OtherAbnormal (supraventricular + fusion grouped together)

The test set remains patient-disjoint (DS2). Training data are balanced only within the training set. The website uses the primary model first and only displays the subtype as secondary information.

This is still an educational/research prototype; the model is not clinically validated.
