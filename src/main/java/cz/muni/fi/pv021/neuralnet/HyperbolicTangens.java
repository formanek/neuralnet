package cz.muni.fi.pv021.neuralnet;

import java.util.function.UnaryOperator;

/**
 * Parametrized hyperbolic tangens activation function
 *
 * @author David Formanek
 */
public class HyperbolicTangens implements UnaryOperator<Double> {

    private final double a;
    private final double b;

    /**
     * Constructs function f(t) = a * tanh(b * t)
     *
     * @param a parameter a
     * @param b parameter b
     */
    public HyperbolicTangens(double a, double b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public Double apply(Double t) {
        return a * Math.tanh(b * t);
    }
}
