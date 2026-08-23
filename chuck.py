import json

raw_domain_corpus = """
Chemical Engineering Thermodynamics and Reaction Kinetics Fundamentals.

Chapter 1: Non-Ideal Gas Behavior and Equations of State.
In chemical process engineering, real gases deviate significantly from ideal gas behavior under elevated pressures and low temperatures. The Peng-Robinson Equation of State (PR-EOS) is widely applied to model vapor-liquid equilibria (VLE) for non-polar and weakly polar hydrocarbon mixtures. The PR-EOS relates pressure P, molar volume V, and temperature T via the formulation: P = (R*T)/(V - b) - a(T)/(V^2 + 2*b*V - b^2), where parameters a(T) and b account for attractive intermolecular forces and co-volume molecular dimensions, respectively. For highly polar systems involving ethanol, water, and organic acids, activity coefficient models such as NRTL (Non-Random Two-Liquid) and UNIQUAC are preferred to account for liquid phase non-idealities and azeotropic behavior.

Chapter 2: Heterogeneous Catalysis and Packed Bed Reactor Modeling.
Packed Bed Reactors (PBR) containing heterogeneous solid catalysts represent the workhorse of industrial chemical synthesis, including Steam Methane Reforming (SMR), Fischer-Tropsch synthesis, and ammonia synthesis. Catalytic activity is governed by reactant adsorption onto active catalytic sites, surface chemical reaction, and product desorption. The kinetic rate equation for Langmuir-Hinshelwood-Hougen-Watson (LHHW) mechanisms accounts for competitive adsorption of chemical species. Temperature dependence of the intrinsic reaction rate constant follows the Arrhenius relationship k(T) = A * exp(-Ea / (R*T)), where A is the pre-exponential factor and Ea is the apparent activation energy in kJ/mol.
"""

# Split raw corpus into text chunks for Causal Language Modeling
text_chunks = [p.strip() for p in raw_domain_corpus.split("\n\n") if p.strip()]

# Save to JSONL with a single 'text' field
with open("dapt_corpus.jsonl", "w", encoding="utf-8") as f:
    for chunk in text_chunks:
        f.write(json.dumps({"text": chunk}) + "\n")

print(f"✅ Prepared DAPT corpus with {len(text_chunks)} raw text blocks!")
