using System;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private int[] structure;
        private double[][][] weights; // weights[layer][neuron][weight]
        private double[][] biases;    // biases[layer][neuron]
        private double[][] outputs;   // outputs[layer][neuron]
        private double[][] errors;    // errors[layer][neuron]

        private Random random = new Random();
        private double learningRate = 0.1;

        public StudentNetwork(int[] structure)
        {
            this.structure = structure;
            InitializeNetwork();
        }

        private void InitializeNetwork()
        {
            int layers = structure.Length;

            // Инициализация массивов
            weights = new double[layers - 1][][];
            biases = new double[layers - 1][];
            outputs = new double[layers][];
            errors = new double[layers][];

            // Инициализация для входного слоя
            outputs[0] = new double[structure[0]];

            // Инициализация для скрытых и выходных слоев
            for (int layer = 1; layer < layers; layer++)
            {
                int neurons = structure[layer];
                int prevNeurons = structure[layer - 1];

                weights[layer - 1] = new double[neurons][];
                biases[layer - 1] = new double[neurons];
                outputs[layer] = new double[neurons];
                errors[layer] = new double[neurons];

                // Инициализация весов случайными значениями
                double range = Math.Sqrt(6.0 / (prevNeurons + neurons));

                for (int neuron = 0; neuron < neurons; neuron++)
                {
                    weights[layer - 1][neuron] = new double[prevNeurons];
                    for (int w = 0; w < prevNeurons; w++)
                    {
                        // Инициализация по методу Xavier/Glorot
                        weights[layer - 1][neuron][w] = (random.NextDouble() * 2 * range) - range;
                    }
                    biases[layer - 1][neuron] = (random.NextDouble() * 2 * range) - range;
                }
            }
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            return x * (1 - x);
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iterations = 0;
            double error = double.MaxValue;

            while (error > acceptableError && iterations < 10000)
            {
                // Прямое распространение
                Forward(sample.input, parallel);

                // Обратное распространение ошибки
                Backward(sample.Output, parallel);

                // Обновление весов
                UpdateWeights(sample.input, learningRate, parallel);

                // Вычисление ошибки
                sample.ProcessPrediction(outputs[structure.Length - 1]);
                error = sample.EstimatedError();
                iterations++;
            }

            return iterations;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            double error = double.MaxValue;
            int epoch = 0;

            while (epoch < epochsCount && error > acceptableError)
            {
                error = 0;

                // Перемешиваем данные для каждой эпохи
                for (int i = 0; i < samplesSet.Count; i++)
                {
                    int j = random.Next(i, samplesSet.Count);
                    var temp = samplesSet[i];
                    samplesSet[i] = samplesSet[j];
                    samplesSet[j] = temp;
                }

                // Обучение на каждом образце
                for (int i = 0; i < samplesSet.Count; i++)
                {
                    var sample = samplesSet[i];

                    // Прямое распространение
                    Forward(sample.input, parallel);

                    // Обратное распространение ошибки
                    Backward(sample.Output, parallel);

                    // Обновление весов
                    UpdateWeights(sample.input, learningRate, parallel);

                    // Вычисление ошибки для этого образца
                    sample.ProcessPrediction(outputs[structure.Length - 1]);
                    error += sample.EstimatedError();
                }

                error /= samplesSet.Count;
                epoch++;

                // Обновление прогресса
                OnTrainProgress((double)epoch / epochsCount, error, TimeSpan.Zero);
            }

            return error;
        }

        protected override double[] Compute(double[] input)
        {
            Forward(input, false);
            return outputs[structure.Length - 1];
        }

        private void Forward(double[] input, bool parallel)
        {
            // Установка входных значений
            Array.Copy(input, outputs[0], input.Length);

            // Прямое распространение через скрытые слои
            for (int layer = 1; layer < structure.Length; layer++)
            {
                int layerIndex = layer - 1;

                if (parallel)
                {
                    Parallel.For(0, structure[layer], neuron =>
                    {
                        double sum = biases[layerIndex][neuron];

                        for (int prevNeuron = 0; prevNeuron < structure[layer - 1]; prevNeuron++)
                        {
                            sum += weights[layerIndex][neuron][prevNeuron] * outputs[layer - 1][prevNeuron];
                        }

                        outputs[layer][neuron] = Sigmoid(sum);
                    });
                }
                else
                {
                    for (int neuron = 0; neuron < structure[layer]; neuron++)
                    {
                        double sum = biases[layerIndex][neuron];

                        for (int prevNeuron = 0; prevNeuron < structure[layer - 1]; prevNeuron++)
                        {
                            sum += weights[layerIndex][neuron][prevNeuron] * outputs[layer - 1][prevNeuron];
                        }

                        outputs[layer][neuron] = Sigmoid(sum);
                    }
                }
            }
        }

        private void Backward(double[] target, bool parallel)
        {
            int lastLayer = structure.Length - 1;

            // Вычисление ошибок для выходного слоя
            if (parallel)
            {
                Parallel.For(0, structure[lastLayer], neuron =>
                {
                    double output = outputs[lastLayer][neuron];
                    errors[lastLayer][neuron] = (target[neuron] - output) * SigmoidDerivative(output);
                });
            }
            else
            {
                for (int neuron = 0; neuron < structure[lastLayer]; neuron++)
                {
                    double output = outputs[lastLayer][neuron];
                    errors[lastLayer][neuron] = (target[neuron] - output) * SigmoidDerivative(output);
                }
            }

            // Обратное распространение ошибки через скрытые слои
            for (int layer = lastLayer - 1; layer > 0; layer--)
            {
                int layerIndex = layer - 1;

                if (parallel)
                {
                    Parallel.For(0, structure[layer], neuron =>
                    {
                        double errorSum = 0;

                        for (int nextNeuron = 0; nextNeuron < structure[layer + 1]; nextNeuron++)
                        {
                            errorSum += errors[layer + 1][nextNeuron] * weights[layer][nextNeuron][neuron];
                        }

                        errors[layer][neuron] = errorSum * SigmoidDerivative(outputs[layer][neuron]);
                    });
                }
                else
                {
                    for (int neuron = 0; neuron < structure[layer]; neuron++)
                    {
                        double errorSum = 0;

                        for (int nextNeuron = 0; nextNeuron < structure[layer + 1]; nextNeuron++)
                        {
                            errorSum += errors[layer + 1][nextNeuron] * weights[layer][nextNeuron][neuron];
                        }

                        errors[layer][neuron] = errorSum * SigmoidDerivative(outputs[layer][neuron]);
                    }
                }
            }
        }

        private void UpdateWeights(double[] input, double learningRate, bool parallel)
        {
            // Обновление весов для первого скрытого слоя
            int firstHiddenLayer = 0;
            if (parallel)
            {
                Parallel.For(0, structure[1], neuron =>
                {
                    for (int prevNeuron = 0; prevNeuron < structure[0]; prevNeuron++)
                    {
                        weights[firstHiddenLayer][neuron][prevNeuron] +=
                            learningRate * errors[1][neuron] * input[prevNeuron];
                    }
                    biases[firstHiddenLayer][neuron] += learningRate * errors[1][neuron];
                });
            }
            else
            {
                for (int neuron = 0; neuron < structure[1]; neuron++)
                {
                    for (int prevNeuron = 0; prevNeuron < structure[0]; prevNeuron++)
                    {
                        weights[firstHiddenLayer][neuron][prevNeuron] +=
                            learningRate * errors[1][neuron] * input[prevNeuron];
                    }
                    biases[firstHiddenLayer][neuron] += learningRate * errors[1][neuron];
                }
            }

            // Обновление весов для остальных слоев
            for (int layer = 2; layer < structure.Length; layer++)
            {
                int layerIndex = layer - 1;

                if (parallel)
                {
                    Parallel.For(0, structure[layer], neuron =>
                    {
                        for (int prevNeuron = 0; prevNeuron < structure[layer - 1]; prevNeuron++)
                        {
                            weights[layerIndex][neuron][prevNeuron] +=
                                learningRate * errors[layer][neuron] * outputs[layer - 1][prevNeuron];
                        }
                        biases[layerIndex][neuron] += learningRate * errors[layer][neuron];
                    });
                }
                else
                {
                    for (int neuron = 0; neuron < structure[layer]; neuron++)
                    {
                        for (int prevNeuron = 0; prevNeuron < structure[layer - 1]; prevNeuron++)
                        {
                            weights[layerIndex][neuron][prevNeuron] +=
                                learningRate * errors[layer][neuron] * outputs[layer - 1][prevNeuron];
                        }
                        biases[layerIndex][neuron] += learningRate * errors[layer][neuron];
                    }
                }
            }
        }
    }
}