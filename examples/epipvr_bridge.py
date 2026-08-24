from cropmix.epipvr import AccessPeriodAssay, AccessPeriodExperiment, EpiPvrBackend

backend = EpiPvrBackend()
print(backend.check_installation())

# Replace these demonstration counts with the real access-period experiment.
aap = AccessPeriodAssay((2, 4, 6), (30, 30, 30), (10, 18, 21))
iap = AccessPeriodAssay((0.25, 0.5, 1.0), (30, 30, 30), (5, 12, 20))
experiment = AccessPeriodExperiment.spt(
    acquisition=aap,
    inoculation=iap,
    fixed_inoculation_for_acquisition=6,
    fixed_acquisition_for_inoculation=4,
    vectors_per_plant=20,
)

# fit = backend.fit(experiment)
# print(fit.parameter_summary())
