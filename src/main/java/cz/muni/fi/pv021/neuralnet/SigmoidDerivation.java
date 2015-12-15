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
public class SigmoidDerivation implements UnaryOperator<Double> {

    private final double lam;

    /**
     * Constructs function f(t) = a * tanh(b * t)
     *
     * @param a parameter a
     * @param b parameter b
     */
    public SigmoidDerivation(double lam) {
        this.lam = lam;
    }

    SigmoidDerivation(double d, double d0) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Double apply(Double y) {
        //return b / a * (a - y) * (a + y);
        return this.lam * y * (1 - y);
    }
}
