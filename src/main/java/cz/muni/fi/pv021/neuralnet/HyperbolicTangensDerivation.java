/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.muni.fi.pv021.neuralnet;

import java.util.function.UnaryOperator;

/**
 *
 * @author Mirek
 */
public class HyperbolicTangensDerivation implements UnaryOperator<Double> {

    private final double a;
    private final double b;

    /**
     * Constructs function f(t) = a * tanh(b * t)
     *
     * @param a parameter a
     * @param b parameter b
     */
    public HyperbolicTangensDerivation(double a, double b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public Double apply(Double y) {
        return b / a * (a - y) * (a + y);
        //return (4 * this.a * this.b * Math.pow(Math.cosh(this.b * y), 2)) /(1 + Math.pow(Math.cosh(2 * this.b * y), 2));
    }
}
