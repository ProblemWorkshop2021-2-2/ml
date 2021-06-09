package org.deeplearning4j.examples.sample;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.util.Arrays.asList;

public class LinearRegression {

    static Double predictForValue(int predictForDependentVariable) {

        BufferedReader reader;
        ArrayList<Integer> x = new ArrayList<>();
        ArrayList<Integer> y = new ArrayList<>();
        try {
            reader = new BufferedReader(new FileReader(
                    "C:\\Users\\kwozn\\Documents\\Projekty\\ml\\dummy_regression.csv"));
            String line = reader.readLine();
            while (line != null) {
                String[] data = line.split(",");
                x.add(Integer.parseInt(data[0]));
                y.add(Integer.parseInt(data[1]));
                line = reader.readLine();
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println(x);
        System.out.println(y);

        if (x.size() != y.size())
            throw new IllegalStateException("Must have equal X and Y data points");


        Integer numberOfDataValues = x.size();

        List<Double> xSquared = x
                .stream()
                .map(position -> Math.pow(position, 2))
                .collect(Collectors.toList());

        List<Integer> xMultipliedByY = IntStream.range(0, numberOfDataValues)
                .map(i -> x.get(i) * y.get(i))
                .boxed()
                .collect(Collectors.toList());

        Integer xSummed = x
                .stream()
                .reduce((prev, next) -> prev + next)
                .get();

        Integer ySummed = y
                .stream()
                .reduce((prev, next) -> prev + next)
                .get();

        Double sumOfXSquared = xSquared
                .stream()
                .reduce((prev, next) -> prev + next)
                .get();

        Integer sumOfXMultipliedByY = xMultipliedByY
                .stream()
                .reduce((prev, next) -> prev + next)
                .get();

        int slopeNominator = numberOfDataValues * sumOfXMultipliedByY - ySummed * xSummed;
        Double slopeDenominator = numberOfDataValues * sumOfXSquared - Math.pow(xSummed, 2);
        Double slope = slopeNominator / slopeDenominator;

        double interceptNominator = ySummed - slope * xSummed;
        double interceptDenominator = numberOfDataValues;
        Double intercept = interceptNominator / interceptDenominator;
        System.out.println("y = "+ slope.toString() + "x" + " + " + intercept.toString());
        return (slope * predictForDependentVariable) + intercept;
    }

    public static void main(String[] args) {
        System.out.println(predictForValue(13));
    }
}