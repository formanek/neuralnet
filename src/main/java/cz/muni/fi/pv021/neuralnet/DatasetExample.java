/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.muni.fi.pv021.neuralnet;

/**
 * Class to hold one example/elemnent of training 
 *
 * @author Mirek
 */
public class DatasetExample {
    double[] inputs;
    double[] outputs;
    
    public DatasetExample(double[] inputs, double[] output) {
        this.inputs = inputs;
        this.outputs = output;
    }
    
    public double[] getInputs() {
        return this.inputs;
    }
    
    public double[] getOutputs() {
        return this.outputs;
    }
    
    @Override
    public String toString() {
        String result = new String("[");
        for (double value : this.inputs) {
            result += value + ", ";
        }
        result = result.substring(0, result.length() - 1) + "|";
        for (double value : this.outputs) {
            result += value + ", ";
        }
        result = result.substring(0, result.length() - 1) + "]";  
        return result;
    }
}
