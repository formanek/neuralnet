/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.muni.fi.pv021.neuralnet.inputData;

import cz.muni.fi.pv021.neuralnet.DatasetExample;
import java.io.IOException;
import java.io.LineNumberReader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Mirek
 */
public class InputDataCSV {
    
    public static List<DatasetExample> getDataFromFile(String inputFilePath) throws IOException {
        //String osAppropriatePath = System.getProperty( "os.name" ).contains( "indow" ) 
        //        ? inputFilePath.substring(1) : inputFilePath;
        Path inputPath = Paths.get(inputFilePath);
        List<DatasetExample> dataset = new ArrayList<>();
        List<String> lines = Files.readAllLines(inputPath, Charset.defaultCharset());
        for (String line : lines) {
            String[] inputValuesStr = line.split(";")[0].split(",");
            double[] inputValues = new double[inputValuesStr.length];
            int i = 0;
            for (String value : inputValuesStr) {
                inputValues[i++] = Double.parseDouble(value);
            }
            String[] outputValuesStr = line.split(";")[1].split(",");
            double[] outputValues = new double[outputValuesStr.length];
            i = 0;
            for (String value : outputValuesStr) {
                outputValues[i++] = Double.parseDouble(value);
            }
            DatasetExample newExample = new DatasetExample(inputValues, outputValues);
            dataset.add(newExample);
        }
        return dataset;
    }
}
