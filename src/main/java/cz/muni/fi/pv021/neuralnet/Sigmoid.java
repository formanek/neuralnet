package cz.muni.fi.pv021.neuralnet;

import java.util.function.UnaryOperator;

/**
 * Parametrized hyperbolic tangens activation function
 *
 * @author David Formanek
 */
public class Sigmoid implements UnaryOperator<Double> {

    private final double lam;

    /**
     * Constructs function f(t) = a * tanh(b * t)
     *
     * @param lam parameter lambda
     */
    public Sigmoid(double lam) {
        this.lam = lam;
    }

    Sigmoid(double d, double d0) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Double apply(Double t) {
        return 1/(1 + Math.exp(-this.lam*t));
    }
}
