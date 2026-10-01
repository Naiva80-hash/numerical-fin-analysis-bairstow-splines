# Numerical Fin Analysis Using Bairstow Root Finding and Cubic Splines

Numerical methods project developed for Advanced Numerical Methods in Chemical Engineering at Sharif University of Technology.

## Overview

This project analyzes the temperature distribution along a variable-geometry aluminum fin whose analytical solution involves a modified Bessel function.

The implementation includes:

- Bessel-series evaluation of the analytical temperature profile
- Bairstow's method for polynomial root finding
- Natural cubic spline interpolation
- Clamped cubic spline interpolation
- Custom LU decomposition and linear-system solution
- Numerical and analytical differentiation
- Error analysis of temperature and derivative approximations

## Technologies

Python, NumPy, pandas, and Matplotlib.

## Run

```bash
pip install -r requirements.txt
python src/main.py
