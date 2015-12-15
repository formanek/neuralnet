/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.muni.fi.pv021.neuralnet;

import java.util.Arrays;
import java.util.List;

/**
 *
 * @author Mirek
 */
public class Trainer {
    private final LayeredNeuralNetwork network;
    private final List<DatasetExample> dataset;
    private double epsilon;
    
    public Trainer(LayeredNeuralNetwork neuralNetwork, List<DatasetExample> dataset, double epsilon) {
        this.network = neuralNetwork;
        this.dataset = dataset;
        this.epsilon = epsilon;
    }
    
    public void train(int maxIterations) {
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double error = 0;
            for (DatasetExample example : this.dataset) {
                double[] inputs = example.getInputs();
                double[] outputs = example.getOutputs();
                if (inputs.length != this.network.getInputLayer().getNeurons().size() 
                        || outputs.length != this.network.getLayers().get(this.network.getLayers()
                                .size() - 1).getNeurons().size()) {
                    System.err.println("Wrong dataset example length");
                    continue;
                }
                //System.out.println("-----------------------------------------------------------------");
                network.setData(inputs);
                //System.out.println("Weights:" + this.network.weightsToString());
                double[] results = network.getResult();
                //System.out.println("NeuronOuts: " + this.network.outputsToString());
                System.out.println("Input: " + Arrays.toString(inputs));
                System.out.println("\tResults: " + Arrays.toString(results));
                System.out.println("\tRef. Outputs: " + Arrays.toString(outputs));
                //delta precomputation
                List<Neuron> lastLayerNeurons = this.network.getLayers().get(this.network.getLayers().size() - 1).getNeurons();
                double errorTmp = 0;
                //System.out.println("Deltas: " + this.network.deltasToString());
                for (int i = 0; i < lastLayerNeurons.size(); i++) {
                    lastLayerNeurons.get(i).computeLastLayerDelta(outputs[i]);
                    errorTmp += Math.pow(outputs[i] - results[i], 2); 
                            //- lastLayerNeurons.get(i).getFunctionOutput(), 2);
                }
                error += 0.5 * errorTmp;
                
                for(int j = this.network.getLayers().size() - 2; j >= 0; j--) {
                    for(Neuron neuron : this.network.getLayers().get(j).getNeurons()) {
                        neuron.computeDelta();
                    }
                }
                //System.out.println("Deltas: " + this.network.deltasToString());
                //weights adaptation
                for (NeuralNetworkLayer layer : this.network.getLayers()) {
                    for (Neuron neuron : layer.getNeurons()) {
                        for (NeuralConnection connection : neuron.getInputs()){ 
                            connection.incrementWeightChange(this.epsilon);
                        }
                    }
                }
                //System.out.println("WeightChanges: " + this.network.weightChangesToString());
            for (NeuralNetworkLayer layer : this.network.getLayers()) {
                for (Neuron neuron : layer.getNeurons()) {
                    for (NeuralConnection connection : neuron.getInputs()){ 
                        connection.adaptWeight(this.epsilon);
                        connection.clearWeightChange();
                    }
                }
            }            
            }

            //System.out.println("new weights:" + this.network.weightsToString());
            System.out.println("....::::Iteration: " + iteration + ": Training error: " + error + "::::....");
        }
    }
    
}
