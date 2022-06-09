### A Pluto.jl notebook ###
# v0.19.5

#> [frontmatter]

using Markdown
using InteractiveUtils

# ╔═╡ c15d5f71-2c67-45b5-b9d3-9d93b8934ca7
begin
	using Pkg
	Pkg.activate(pwd())
	# Pkg.add(["CSV", "DataFrames"])
	# Pkg.add("StatsBase")
	# Pkg.add("Pipe")
	# Pkg.add("MLDataUtils")
	# Pkg.add("MLJ")
	# Pkg.add("StatsPlots")
	# Pkg.add("KernelDensity")
	# Pkg.add("WGLMakie")
	# # Pkg.rm("WGLMakie")
	# Pkg.add("StableRNGs")
	# Pkg.add(name="Lighthouse", version="0.14")
	# Pkg.add("PlutoUI")
	# Pkg.update()
	
	using Plots, StatsPlots, PlutoUI
	using KernelDensity
	using WGLMakie
	using DataFrames, CSV
	using Statistics, StatsBase, StableRNGs
	using Pipe
	using MLDataUtils, MLJ, Lighthouse
end

# ╔═╡ aa737c92-e1a1-11ec-343e-116d4ead64c3
iris = @pipe CSV.read(download("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), DataFrame, header=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]) |> coerce!(_, :species => Multiclass)

# ╔═╡ b5078a27-d278-4106-9daf-4f498ad48c72
schema(iris)

# ╔═╡ dcb4f3b7-6885-43dc-91c6-d4a0b300aa36
isnothing.(iris) |> Matrix |> sum

# ╔═╡ 77d6184e-590c-4db5-88eb-779e0a7f5d87
y_col = "species"

# ╔═╡ 8e692d0d-face-43b3-9ee8-9bfb2cee3ab1
md"splitting the dataset"

# ╔═╡ 35545a54-f86a-4ee7-97d8-2f927b656b95
train, test = stratifiedobs(row->row[:species], iris); nothing

# ╔═╡ eff06dfa-a7a1-478e-b8cd-2a3d8215b295
md"we don't look at testset"

# ╔═╡ b97d95df-31ee-4ad3-9748-2bf2f65f909f
@pipe describe(train, :all) |> permutedims(_, 1)

# ╔═╡ 0f4887cc-7e82-4db1-9f87-b4da7fe7a03f
md"but, it's okay to look at the population distribution of the entire dataset(including testset)"

# ╔═╡ dddf10c3-b8de-48fa-be8d-8c155b7f94b7
let
	species_dist = countmap(iris.species)
	species_dist = convert(Dict{eltype(keys(species_dist)), Float64}, species_dist)
	for (species, count) ∈ species_dist
		species_dist[species] /= size(iris, 1)
	end
	bar(species_dist)
end

# ╔═╡ 4b6c1306-63bb-42df-9e57-13ebec4f42cd
md"population is normally distributed: no target imbalance"

# ╔═╡ 4949e3ac-2432-4bcd-aeee-bcd29dea23dc
md"# EDA"

# ╔═╡ 6097ca7d-97b3-4691-9a47-a66f6105a732
begin
	features = names(train)[1:end-1]
	n = length(features)
	ps = fill(Plots.plot(), n, n)
	
	for i ∈ 1:n, j ∈ 1:n
		x, y = train[!, features[i]], train[!, features[j]]
		species = train[!, :species]
		ps[i, j] = if j < i
			dens = KernelDensity.kde((x, y))
			Plots.plot(dens)
		elseif j > i
			Plots.plot(x, y, seriestype=:scatter, group=species)
		else
			Plots.plot(x, y, seriestype=:density, group=species)
		end
		i == 1 && Plots.ylabel!(ps[i, j], features[j])
		j == n && Plots.xlabel!(ps[i, j], features[i])
	end
	Plots.plot(ps..., layout = (n, n), size=(1000,1000))
end

# ╔═╡ faed050a-c26a-4083-8822-ea6763e17645
md"""
we find that:
* can't see any significant outliers that would skew the distributions
* Because Patel length and Petal width have saperable distributions for different iris species, these 2 variables would have higher influence over classifying the species.
* We can also observe from the scatterplots of petal_length x petal_width, petal_width x sepal_width and petal_width x sepal_length, that the species clusters are clearly distinguishable and can easily be saperated a line.
* Hence, a sinple linear model(s) would be best suited for this type of classification here, without requiring of any feature transformations as the features are already distinguishable.
* The setosa species is the most easily distinguishable because of its small feature size.
"""

# ╔═╡ 5db70ee8-a94c-4acc-80dc-9ba28521ae70
function plot_heatmap(mat; xlabel)
	Plots.heatmap(mat, 
		xticks=(1:length(xlabel), xlabel), yticks=(1:length(xlabel), xlabel), 
	    aspect_ratio=:equal, )
	
	fontsize = 15
	nrow, ncol = size(mat)
	ann = [(i,j, Plots.text(round(mat[i,j], digits=2), fontsize, :white, :center))
	            for i in 1:size(mat, 1) for j in 1:size(mat, 2) if 1 > abs(mat[i,j]) > .5]
	annotate!(ann, linecolor=:white)
end

# ╔═╡ f37d36a8-e96d-4f82-a3f1-f8b9180af37d
@pipe cor(Matrix(train[!, Not(y_col)])) |> plot_heatmap(_, xlabel=names(train)[1:end-1])

# ╔═╡ a68185e2-1de0-4714-b198-0b2a503e0bec
md"* if we look further we can see that `petal_width` is highly +vely correlated to `petal_length` and `sepal_length`. Similarly, `petal_length` and `sepal_length` are also sognificantly correlated.
* Also, `sepal_width` is slightly -vely correlated with other variables."

# ╔═╡ aa8bc163-8363-4b70-9920-c77818108e59
@pipe cov(Matrix(train[!, Not(y_col)])) |> plot_heatmap(_, xlabel=names(train)[1:end-1])

# ╔═╡ 0f9f1644-3ee0-4595-9b8d-96067d398312
md"A strong correlation is present between petal width and petal length.
"

# ╔═╡ b4e66962-2f9f-4402-9309-fd5701b2995a
md"# Modeling

we earlier found out that the dataset is not too complex and simple models(most probably linear models) would be better suited for this problem."

# ╔═╡ 8f469532-73b3-452c-95af-7544d9c4d84f
begin
	X, y = iris[!, Not(y_col)], iris[!, y_col]
	(X_train, y_train), (X_test, y_test) = stratifiedobs((X, y))
	size(X_train), size(y_train), size(X_test), size(y_test)
end

# ╔═╡ 3cf1c83e-12dd-441f-9257-efcbbc9d8a89
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels add=true

# ╔═╡ 9eb53a31-0b7f-43d6-b30c-dc2fb790f4aa
pipe = @pipeline LogisticClassifier

# ╔═╡ d99cebb3-093f-438f-b9de-ff3dca7ca521
model = machine(pipe, X_train, y_train)

# ╔═╡ 73237751-34a1-4035-b2c6-79682ba684ac
MLJ.evaluate!(model, resampling=CV(nfolds=6, rng=StableRNG(32)),
			measures=[MLJ.accuracy, log_loss])

# ╔═╡ d9188017-22f6-4aad-be40-5c10273070ea
md"## Model Tuning"

# ╔═╡ b5fc3433-019c-401b-aedb-56fa07c6c6a5
ranges = [range(pipe, :(logistic_classifier.lambda), lower=0.0, upper=1.0, scale=:log),
		range(pipe, :(logistic_classifier.gamma), lower=0.0, upper=1.0, scale=:log),
		range(pipe, :(logistic_classifier.penalty), values=[:l2, :l1, :en, :none]),
]

# ╔═╡ c4e17f84-afba-4650-92f4-c0631d9d0ba7
tm = TunedModel(model=pipe, ranges=ranges, measure=MLJ.accuracy)

# ╔═╡ cafeae07-6097-41db-934e-7c70a30e68c3
tuned_model = machine(tm, X_train, y_train)

# ╔═╡ c838ab8c-f684-47df-b121-7bcd4c707f0e
MLJ.evaluate!(tuned_model, resampling=CV(nfolds=6, rng=StableRNG(32)),
			measures=[MLJ.accuracy, log_loss])

# ╔═╡ c55b0c7e-3960-4fdf-a193-dc2852ec7f23
fitted_params(tuned_model).best_model

# ╔═╡ 29cb56fa-7e45-4287-bc23-36738c9d180f
r = report(tuned_model)

# ╔═╡ 292d9da7-e237-439f-9bec-4261dfe0561c
r.best_history_entry.measurement[1]

# ╔═╡ b8932352-7257-47ea-b5b0-8b51f41da42b
ŷ = MLJ.predict(tuned_model, X_test) .|> mode

# ╔═╡ bf4bc29e-3804-427e-8d68-b7bc6d3b5b0b
Accuracy()(ŷ, y_test)

# ╔═╡ 20cf436a-2982-4d3a-a0e0-337daa95bc4b
begin
	cnf_mat = ConfusionMatrix()(ŷ, y_test)
	Lighthouse.plot_confusion_matrix(cnf_mat.mat, cnf_mat.labels, :Row)
end

# ╔═╡ c87746cf-1f76-488f-b350-123b59642bb4
md"todo: classification report"

# ╔═╡ cc618da9-71a6-4e59-b49a-967834368925
WGLMakie.activate!()

# ╔═╡ Cell order:
# ╠═aa737c92-e1a1-11ec-343e-116d4ead64c3
# ╠═b5078a27-d278-4106-9daf-4f498ad48c72
# ╠═dcb4f3b7-6885-43dc-91c6-d4a0b300aa36
# ╠═77d6184e-590c-4db5-88eb-779e0a7f5d87
# ╟─8e692d0d-face-43b3-9ee8-9bfb2cee3ab1
# ╠═35545a54-f86a-4ee7-97d8-2f927b656b95
# ╟─eff06dfa-a7a1-478e-b8cd-2a3d8215b295
# ╠═b97d95df-31ee-4ad3-9748-2bf2f65f909f
# ╟─0f4887cc-7e82-4db1-9f87-b4da7fe7a03f
# ╠═dddf10c3-b8de-48fa-be8d-8c155b7f94b7
# ╟─4b6c1306-63bb-42df-9e57-13ebec4f42cd
# ╟─4949e3ac-2432-4bcd-aeee-bcd29dea23dc
# ╠═6097ca7d-97b3-4691-9a47-a66f6105a732
# ╟─faed050a-c26a-4083-8822-ea6763e17645
# ╟─5db70ee8-a94c-4acc-80dc-9ba28521ae70
# ╠═f37d36a8-e96d-4f82-a3f1-f8b9180af37d
# ╟─a68185e2-1de0-4714-b198-0b2a503e0bec
# ╠═aa8bc163-8363-4b70-9920-c77818108e59
# ╟─0f9f1644-3ee0-4595-9b8d-96067d398312
# ╟─b4e66962-2f9f-4402-9309-fd5701b2995a
# ╠═8f469532-73b3-452c-95af-7544d9c4d84f
# ╠═3cf1c83e-12dd-441f-9257-efcbbc9d8a89
# ╠═9eb53a31-0b7f-43d6-b30c-dc2fb790f4aa
# ╠═d99cebb3-093f-438f-b9de-ff3dca7ca521
# ╠═73237751-34a1-4035-b2c6-79682ba684ac
# ╟─d9188017-22f6-4aad-be40-5c10273070ea
# ╠═b5fc3433-019c-401b-aedb-56fa07c6c6a5
# ╟─c4e17f84-afba-4650-92f4-c0631d9d0ba7
# ╠═cafeae07-6097-41db-934e-7c70a30e68c3
# ╠═c838ab8c-f684-47df-b121-7bcd4c707f0e
# ╠═c55b0c7e-3960-4fdf-a193-dc2852ec7f23
# ╠═29cb56fa-7e45-4287-bc23-36738c9d180f
# ╠═292d9da7-e237-439f-9bec-4261dfe0561c
# ╠═b8932352-7257-47ea-b5b0-8b51f41da42b
# ╠═bf4bc29e-3804-427e-8d68-b7bc6d3b5b0b
# ╠═20cf436a-2982-4d3a-a0e0-337daa95bc4b
# ╟─c87746cf-1f76-488f-b350-123b59642bb4
# ╠═cc618da9-71a6-4e59-b49a-967834368925
# ╠═c15d5f71-2c67-45b5-b9d3-9d93b8934ca7
