using RxInfer
using Plots
using CSV
using DataFrames


path = abspath("C:\\Users\\lkiel\\PycharmProjects\\RobustnessMetric\\Datasets\\threshold_distributions_fairness")

if !isfile(joinpath(path, "df_epsilon_fairness.csv")) || !isfile(joinpath(path, "df_epsilon_crit_fairness.csv"))
    error("df of epsilon fairness not found, please run python import first to generate it!")
end

df_epsilon = DataFrame(CSV.File(joinpath(path, "df_epsilon_fairness.csv")))
df_epsilon_crit = DataFrame(CSV.File(joinpath(path, "df_epsilon_crit_fairness.csv")))

Networks = unique(df_epsilon_crit.network)

for network in Networks:
    # take only test data
    df_epsilon_network = df_epsilon[df_epsilon.ds .== "test", :]

    # get relevant data for that network
    df_epsilon_network = df_epsilon_network[df_epsilon_network.network .== network, [:epsilon,:result,:image,:runtime]]

    # make a new dataframe with the lower and upper bound for each image
    df_bounds = DataFrame(image = unique(df_epsilon_network.image), lower_bound = 0.0, upper_bound = 0.4)

    # get upper and lower bound for each image
    for img in unique(df_epsilon_network.image)
        epsilon_for_image = @view df_epsilon_network[df_epsilon_network.image .== img, :]
        unsat_epsilons = epsilon_for_image[epsilon_for_image.result .== "unsat", :].epsilon
        sat_epsilons = epsilon_for_image[epsilon_for_image.result .== "sat", :].epsilon

        row_index = findfirst(df_bounds.image .== img)

        if !isempty(unsat_epsilons)
            df_bounds[row_index, :lower_bound] = maximum(unsat_epsilons)
        else
            println("empty, unsat", ' ', img)
        end

        if !isempty(sat_epsilons)
            df_bounds[row_index, :upper_bound] = minimum(sat_epsilons)
        else
            println("empty, sat", ' ', img)
        end
    end
end