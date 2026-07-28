# Fixed-abs high-dimensional tuning protocol

Candidates: `96` (6 scenarios × 4 methods × 4 candidates × seed 0).
Tuning budget: `150` communication rounds per candidate.

Selection is performed separately for every scenario and method.
Only validation metrics may rank candidates. Candidates with numerical divergence are excluded.
Primary key: lowest `best_validation_mse`; tie-break by lower last-50 validation MSE standard deviation,
then lower final-versus-best validation gap. Test MSE is reported only after the candidate is fixed.

FedGDA and FedOGDA receive identical candidate counts and grids within each stochasticity regime.
The final five-seed runs use the selected candidate with the original 500-round budget.
