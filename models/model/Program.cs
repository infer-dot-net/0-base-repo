using System;
using Microsoft.ML.Probabilistic.Models;
using D = Microsoft.ML.Probabilistic.Distributions;
using A = Microsoft.ML.Probabilistic.Algorithms;
using System.IO;
using System.Text;
namespace model
{
    class Program
    {
        static void Main(string[] args)
        {

            /********* arguments **********/
            string dataDir = args[0];
            string coinTrue = args[1];
            /******************************/

            /********* model setup **********/
            Variable<bool> firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
            Variable<bool> secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
            Variable<bool> bothHeads = (firstCoin & secondCoin).Named("bothHeads");
            /********************************/

            /****** inference engine *******/
            InferenceEngine engine = new InferenceEngine(new A.GibbsSampling());
            engine.ShowFactorGraph = true;
            /*******************************/


            /********* prior observation ********/
            D.Bernoulli priorFirstCoin = engine.Infer<D.Bernoulli>(firstCoin);
            D.Bernoulli priorSecondCoin = engine.Infer<D.Bernoulli>(secondCoin);
            D.Bernoulli priorBothHeads = engine.Infer<D.Bernoulli>(bothHeads);
            /************************************/

            /********* observations *********/
            if (coinTrue == "first")
            {
                firstCoin.ObservedValue = true;
            }
            else if (coinTrue == "second")
            {
                secondCoin.ObservedValue = true;
            }
            else if (coinTrue == "both")
            {
                bothHeads.ObservedValue = true;
            }
            /*******************************/


            /********* post observation *********/
            D.Bernoulli postFirstCoin = engine.Infer<D.Bernoulli>(firstCoin);
            D.Bernoulli postSecondCoin = engine.Infer<D.Bernoulli>(secondCoin);
            D.Bernoulli postBothHeads = engine.Infer<D.Bernoulli>(bothHeads);
            /************************************/


            /********* print outs *********/
            Console.WriteLine("\npriorFirstCoin : {0}",priorFirstCoin);
            Console.WriteLine("priorSecondCoin: {0}",priorSecondCoin);
            Console.WriteLine("priorBothHeads : {0}",priorBothHeads);

            Console.WriteLine("\npostFirstCoin  : {0}",postFirstCoin);
            Console.WriteLine("postSecondCoin : {0}",postSecondCoin);
            Console.WriteLine("postBothHeads  : {0}",postBothHeads);
            /*******************************/

            


            /******* creating results.csv ******/
            var results = new StringBuilder();

            results.AppendLine("probability;variable;mean");
            var line = string.Format("prior;firstCoin;{0}", priorFirstCoin.GetMean());
            results.AppendLine(line.Replace(',', '.'));
            line = string.Format("prior;secondCoin;{0}", priorSecondCoin.GetMean());
            results.AppendLine(line.Replace(',', '.'));
            line = string.Format("prior;bothHeads;{0}", priorBothHeads.GetMean());
            results.AppendLine(line.Replace(',', '.'));

            line = string.Format("posterior;firstCoin;{0}", postFirstCoin.GetMean());
            results.AppendLine(line.Replace(',', '.'));
            line = string.Format("posterior;secondCoin;{0}", postSecondCoin.GetMean());
            results.AppendLine(line.Replace(',', '.'));
            line = string.Format("posterior;bothHeads;{0}", postBothHeads.GetMean());
            results.AppendLine(line.Replace(',', '.'));

            File.WriteAllText(dataDir + "results.csv", results.ToString());
            /*********************************/

        }
    }
}
