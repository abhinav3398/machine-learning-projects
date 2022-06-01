### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ c15d5f71-2c67-45b5-b9d3-9d93b8934ca7
begin
	using Pkg
	Pkg.activate(pwd())
	Pkg.add("Images")
	Pkg.add("Plots")
	Pkg.add("PlutoUI")
	Pkg.add("ImageBinarization")
	Pkg.add("Noise")
	Pkg.add("ImageView")

	using Plots, Images, ImageBinarization, Noise, ImageView
	using PlutoUI
	using Statistics
end

# ╔═╡ b54937b2-31f1-4b92-a421-c56374eacb46
using BenchmarkTools

# ╔═╡ 6bb07ad8-8afd-45a4-be08-f5492aa77c46
md"# converting an image to grayscale"

# ╔═╡ 76abfe87-1fd7-4429-82c6-f5b2695b84b6
md"# applying median blur to smoothen an image"

# ╔═╡ 4adb8b78-23f8-4d9b-9c34-78be3c460c8e
md"# Create edge mask by retrieving the edges for cartoon effect by using thresholding technique"

# ╔═╡ 9cbec465-48b2-42ca-9540-e2621ce0d8f3
md"""
# Reduce the color palette
"""

# ╔═╡ 487121a4-39f2-4822-aad1-0d12048de1e1
md"Applying bilateral filter to remove noise and keep edge sharpness as required. It would give a bit blurred and sharpness-reducing effect to the image."

# ╔═╡ a3e9bfc6-ce4e-44db-8d0d-dd18c75c598b
md"# Combine Edge Mask with the Colored Image"

# ╔═╡ fb0ec4ff-3431-4200-aba2-f2a08e53f90f
md"# putting it all togather"

# ╔═╡ d9ca0026-e679-4c2f-938a-e0aee525c7e9
begin
	function cartoonify(img; 
				binarize_window_size=9, binarize_percentage=15,
				quantization_lvl=8,
				σ=1, sharpening_intensity=1)
		img_mask = binarize(img, AdaptiveThreshold(window_size = binarize_window_size, percentage=binarize_percentage))
		
		img_color_noise = quantization(img, quantization_lvl)
	
		img_noise_blurred = imfilter(img_color_noise, Kernel.gaussian(σ))
	
		# sharpened = @. img_color_noise * (1 + sharpening_intensity) + img_noise_blurred * (-intensity)
	
		# map(zip(sharpened, img_binary)) do (pix, mask)
		# 	mask == 1 ? pix : RGB(mask)
		# end
	
		map(zip(img_color_noise, img_noise_blurred, img_mask)) do (noisy_pix, blurred_pix, mask)
			mask == 1 ? 
			(noisy_pix * (1 + sharpening_intensity) + blurred_pix*(-sharpening_intensity)) : 
			RGB(mask)
		end
	end
	# todo: reduce allocation
	function cartoonify!(img; 
				binarize_window_size=9, binarize_percentage=15,
				quantization_lvl=8,
				σ=1, sharpening_intensity=1,
				binarize_buffer=Gray.(img), quantization_buffer=copy(img), imfilter_buffer=similar(img), output_buffer=similar(img))
		img_mask = binarize!(binarize_buffer, AdaptiveThreshold(window_size = binarize_window_size, percentage=binarize_percentage))
		
		img_color_noise = quantization!(quantization_buffer, quantization_lvl)
	
		img_noise_blurred = imfilter!(imfilter_buffer, img_color_noise, Kernel.gaussian(σ))

		map(zip(img_color_noise, img_noise_blurred, img_mask)) do (noisy_pix, blurred_pix, mask)
		mask == 1 ? 
			(noisy_pix * (1 + sharpening_intensity) + blurred_pix*(-sharpening_intensity)) : 
			RGB(mask)
		end
	end
end

# ╔═╡ f64453e7-df80-40ad-9a71-f31dc0f6bdd4
md"# benchmarking & improvements

checking type consistancy"

# ╔═╡ b742cd3d-88f1-4740-8c29-6c29c5aacef9


# ╔═╡ 999f487e-5d6f-4fe9-be87-12f82829f8a9
md"""
resources:

[TDS: Turn Photos into Cartoons Using Python](https://towardsdatascience.com/turn-photos-into-cartoons-using-python-bb1a9f578a7e)\
[dataflair: Cartoonify an Image with OpenCV in Python](https://data-flair.training/blogs/cartoonify-image-opencv-python/)
"""

# ╔═╡ 59402321-3b5f-4dbb-9cb4-888a6ee19bda
function process_raw_camera_data(raw_camera_data)
	# the raw image data is a long byte array, we need to transform it into something
	# more "Julian" - something with more _structure_.
	
	# The encoding of the raw byte stream is:
	# every 4 bytes is a single pixel
	# every pixel has 4 values: Red, Green, Blue, Alpha
	# (we ignore alpha for this notebook)
	
	# So to get the red values for each pixel, we take every 4th value, starting at 
	# the 1st:
	reds_flat = UInt8.(raw_camera_data["data"][1:4:end])
	greens_flat = UInt8.(raw_camera_data["data"][2:4:end])
	blues_flat = UInt8.(raw_camera_data["data"][3:4:end])
	
	# but these are still 1-dimensional arrays, nicknamed 'flat' arrays
	# We will 'reshape' this into 2D arrays:
	
	width = raw_camera_data["width"]
	height = raw_camera_data["height"]
	
	# shuffle and flip to get it in the right shape
	reds = reshape(reds_flat, (width, height))' / 255.0
	greens = reshape(greens_flat, (width, height))' / 255.0
	blues = reshape(blues_flat, (width, height))' / 255.0
	
	# we have our 2D array for each color
	# Let's create a single 2D array, where each value contains the R, G and B value of 
	# that pixel
	
	RGB.(reds, greens, blues)
end

# ╔═╡ 7c8d245b-67b8-4968-bcdc-44119487d293
function camera_input(;max_size=150, default_url="https://i.imgur.com/SUmi94P.png")
"""
<span class="pl-image waiting-for-permission">
<style>
	
	.pl-image.popped-out {
		position: fixed;
		top: 0;
		right: 0;
		z-index: 5;
	}

	.pl-image #video-container {
		width: 250px;
	}

	.pl-image video {
		border-radius: 1rem 1rem 0 0;
	}
	.pl-image.waiting-for-permission #video-container {
		display: none;
	}
	.pl-image #prompt {
		display: none;
	}
	.pl-image.waiting-for-permission #prompt {
		width: 250px;
		height: 200px;
		display: grid;
		place-items: center;
		font-family: monospace;
		font-weight: bold;
		text-decoration: underline;
		cursor: pointer;
		border: 5px dashed rgba(0,0,0,.5);
	}

	.pl-image video {
		display: block;
	}
	.pl-image .bar {
		width: inherit;
		display: flex;
		z-index: 6;
	}
	.pl-image .bar#top {
		position: absolute;
		flex-direction: column;
	}
	
	.pl-image .bar#bottom {
		background: black;
		border-radius: 0 0 1rem 1rem;
	}
	.pl-image .bar button {
		flex: 0 0 auto;
		background: rgba(255,255,255,.8);
		border: none;
		width: 2rem;
		height: 2rem;
		border-radius: 100%;
		cursor: pointer;
		z-index: 7;
	}
	.pl-image .bar button#shutter {
		width: 3rem;
		height: 3rem;
		margin: -1.5rem auto .2rem auto;
	}

	.pl-image video.takepicture {
		animation: pictureflash 200ms linear;
	}

	@keyframes pictureflash {
		0% {
			filter: grayscale(1.0) contrast(2.0);
		}

		100% {
			filter: grayscale(0.0) contrast(1.0);
		}
	}
</style>

	<div id="video-container">
		<div id="top" class="bar">
			<button id="stop" title="Stop video">✖</button>
			<button id="pop-out" title="Pop out/pop in">⏏</button>
		</div>
		<video playsinline autoplay></video>
		<div id="bottom" class="bar">
		<button id="shutter" title="Click to take a picture">📷</button>
		</div>
	</div>
		
	<div id="prompt">
		<span>
		Enable webcam
		</span>
	</div>

<script>
	// based on https://github.com/fonsp/printi-static (by the same author)

	const span = currentScript.parentElement
	const video = span.querySelector("video")
	const popout = span.querySelector("button#pop-out")
	const stop = span.querySelector("button#stop")
	const shutter = span.querySelector("button#shutter")
	const prompt = span.querySelector(".pl-image #prompt")

	const maxsize = $(max_size)

	const send_source = (source, src_width, src_height) => {
		const scale = Math.min(1.0, maxsize / src_width, maxsize / src_height)

		const width = Math.floor(src_width * scale)
		const height = Math.floor(src_height * scale)

		const canvas = html`<canvas width=\${width} height=\${height}>`
		const ctx = canvas.getContext("2d")
		ctx.drawImage(source, 0, 0, width, height)

		span.value = {
			width: width,
			height: height,
			data: ctx.getImageData(0, 0, width, height).data,
		}
		span.dispatchEvent(new CustomEvent("input"))
	}
	
	const clear_camera = () => {
		window.stream.getTracks().forEach(s => s.stop());
		video.srcObject = null;

		span.classList.add("waiting-for-permission");
	}

	prompt.onclick = () => {
		navigator.mediaDevices.getUserMedia({
			audio: false,
			video: {
				facingMode: "environment",
			},
		}).then(function(stream) {

			stream.onend = console.log

			window.stream = stream
			video.srcObject = stream
			window.cameraConnected = true
			video.controls = false
			video.play()
			video.controls = false

			span.classList.remove("waiting-for-permission");

		}).catch(function(error) {
			console.log(error)
		});
	}
	stop.onclick = () => {
		clear_camera()
	}
	popout.onclick = () => {
		span.classList.toggle("popped-out")
	}

	shutter.onclick = () => {
		const cl = video.classList
		cl.remove("takepicture")
		void video.offsetHeight
		cl.add("takepicture")
		video.play()
		video.controls = false
		console.log(video)
		send_source(video, video.videoWidth, video.videoHeight)
	}
	
	
	document.addEventListener("visibilitychange", () => {
		if (document.visibilityState != "visible") {
			clear_camera()
		}
	})


	// Set a default image

	const img = html`<img crossOrigin="anonymous">`

	img.onload = () => {
	console.log("helloo")
		send_source(img, img.width, img.height)
	}
	img.src = "$(default_url)"
	console.log(img)
</script>
</span>
""" |> HTML
end

# ╔═╡ aa737c92-e1a1-11ec-343e-116d4ead64c3
@bind webcam_data1 camera_input()

# ╔═╡ dcb4f3b7-6885-43dc-91c6-d4a0b300aa36
img = process_raw_camera_data(webcam_data1)

# ╔═╡ 70ef3422-b0bf-4746-a2ec-05d913c7a2b1
gray_img = Gray.(img)

# ╔═╡ 40e5201a-71b9-45aa-8d0b-d4a2330c968a
smooth_gray_scale = mapwindow(median, gray_img, (5, 5))

# ╔═╡ 37fe2990-35ba-4b17-beae-235b141109b0
img_binary = binarize(img, AdaptiveThreshold(window_size = 9, percentage=10))

# ╔═╡ cdb20c28-e496-4013-9b30-5df057a12468
img_color_noise = quantization(img, 8)

# ╔═╡ 43d88b04-483e-4b55-b6be-72f23cb9f26e
begin
	gaussian_smoothing = 1
	intensity = 1
	img_noise_blurred = imfilter(img_color_noise, Kernel.gaussian(gaussian_smoothing))
end

# ╔═╡ af518249-d14a-4a25-9e5b-cdfd0c6fbcc8
sharpened = @. img_color_noise * (1 + intensity) + img_noise_blurred * (-intensity)

# ╔═╡ 266bbfbb-7244-4f14-a1cd-e63a4918247d
cartoon = map(zip(sharpened, img_binary)) do (pix, mask)
	mask == 1 ? pix : RGB(mask)
end

# ╔═╡ 50aff4f1-aebd-4af8-bf92-267c5fa07980
cartoonify(img)

# ╔═╡ e86876af-5b41-4da5-a453-f5eb5cb01d75
cartoonify!(img; binarize_buffer=Gray.(img), quantization_buffer=copy(img), imfilter_buffer=similar(img), output_buffer=similar(img))

# ╔═╡ f6d2df3e-07ec-4a82-a63a-2a8c774c4a25
with_terminal() do
	@code_warntype cartoonify(img)
	@code_warntype cartoonify!(img; binarize_buffer=Gray.(img), quantization_buffer=copy(img), imfilter_buffer=similar(img), output_buffer=similar(img))
end

# ╔═╡ 268fe4e7-74d8-4e4f-a079-02378669d0a3
@benchmark cartoonify($img)

# ╔═╡ 44d2d5bd-817e-4aa7-924b-22a55546093e
@benchmark cartoonify!($img; binarize_buffer=$(Gray.(img)), quantization_buffer=$(copy(img)), imfilter_buffer=$(similar(img)), output_buffer=$(similar(img)))

# ╔═╡ Cell order:
# ╠═aa737c92-e1a1-11ec-343e-116d4ead64c3
# ╠═dcb4f3b7-6885-43dc-91c6-d4a0b300aa36
# ╟─6bb07ad8-8afd-45a4-be08-f5492aa77c46
# ╠═70ef3422-b0bf-4746-a2ec-05d913c7a2b1
# ╟─76abfe87-1fd7-4429-82c6-f5b2695b84b6
# ╠═40e5201a-71b9-45aa-8d0b-d4a2330c968a
# ╠═4adb8b78-23f8-4d9b-9c34-78be3c460c8e
# ╠═37fe2990-35ba-4b17-beae-235b141109b0
# ╟─9cbec465-48b2-42ca-9540-e2621ce0d8f3
# ╠═cdb20c28-e496-4013-9b30-5df057a12468
# ╟─487121a4-39f2-4822-aad1-0d12048de1e1
# ╠═43d88b04-483e-4b55-b6be-72f23cb9f26e
# ╠═af518249-d14a-4a25-9e5b-cdfd0c6fbcc8
# ╟─a3e9bfc6-ce4e-44db-8d0d-dd18c75c598b
# ╠═266bbfbb-7244-4f14-a1cd-e63a4918247d
# ╟─fb0ec4ff-3431-4200-aba2-f2a08e53f90f
# ╠═d9ca0026-e679-4c2f-938a-e0aee525c7e9
# ╠═50aff4f1-aebd-4af8-bf92-267c5fa07980
# ╠═e86876af-5b41-4da5-a453-f5eb5cb01d75
# ╟─f64453e7-df80-40ad-9a71-f31dc0f6bdd4
# ╠═f6d2df3e-07ec-4a82-a63a-2a8c774c4a25
# ╠═b54937b2-31f1-4b92-a421-c56374eacb46
# ╠═268fe4e7-74d8-4e4f-a079-02378669d0a3
# ╠═44d2d5bd-817e-4aa7-924b-22a55546093e
# ╠═b742cd3d-88f1-4740-8c29-6c29c5aacef9
# ╟─999f487e-5d6f-4fe9-be87-12f82829f8a9
# ╟─59402321-3b5f-4dbb-9cb4-888a6ee19bda
# ╟─7c8d245b-67b8-4968-bcdc-44119487d293
# ╠═c15d5f71-2c67-45b5-b9d3-9d93b8934ca7
