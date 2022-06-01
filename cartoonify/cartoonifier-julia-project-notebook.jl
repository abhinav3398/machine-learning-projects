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
	# Pkg.add("Images")
	# Pkg.add("Plots")
	# Pkg.add("PlutoUI")
	# Pkg.add("ImageBinarization")

	using Plots, Images, ImageBinarization
	using PlutoUI
	using Statistics
end

# ╔═╡ 6bb07ad8-8afd-45a4-be08-f5492aa77c46
md"# converting an image to grayscale"

# ╔═╡ 76abfe87-1fd7-4429-82c6-f5b2695b84b6
md"# applying median blur to smoothen an image"

# ╔═╡ 4adb8b78-23f8-4d9b-9c34-78be3c460c8e
md"# retrieving the edges for cartoon effect by using thresholding technique"

# ╔═╡ 9cbec465-48b2-42ca-9540-e2621ce0d8f3
md"# applying bilateral filter to remove noise and keep edge sharp as required"

# ╔═╡ cdb20c28-e496-4013-9b30-5df057a12468


# ╔═╡ 50aff4f1-aebd-4af8-bf92-267c5fa07980


# ╔═╡ b742cd3d-88f1-4740-8c29-6c29c5aacef9


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

# ╔═╡ ad153aef-42e0-4ed7-a81b-30f697007b0d
begin
	hsv_img = HSV.(smooth_gray_scale)
	channels = channelview(float.(hsv_img))
	hue_img = channels[1,:,:]
	value_img = channels[3,:,:]
	saturation_img = channels[2,:,:]
	nothing
end

# ╔═╡ 459fa2ae-2a98-4a9c-8435-063270d1a7aa
begin
	mask = zeros(size(hue_img))
	h, s, v = 80, 150, 100
	for ind in eachindex(hue_img)
	    if hue_img[ind] <= h && saturation_img[ind] <= s/255 && value_img[ind] <= v/255
	        mask[ind] = 1
	    end
	end
	binary_img = colorview(Gray, mask)
end

# ╔═╡ 37fe2990-35ba-4b17-beae-235b141109b0
imgb = binarize(img, AdaptiveThreshold(window_size = 9, percentage=10))

# ╔═╡ 8a35e6e1-c166-4cde-aa95-d0db94512879


# ╔═╡ Cell order:
# ╠═aa737c92-e1a1-11ec-343e-116d4ead64c3
# ╠═dcb4f3b7-6885-43dc-91c6-d4a0b300aa36
# ╟─6bb07ad8-8afd-45a4-be08-f5492aa77c46
# ╠═70ef3422-b0bf-4746-a2ec-05d913c7a2b1
# ╟─76abfe87-1fd7-4429-82c6-f5b2695b84b6
# ╠═40e5201a-71b9-45aa-8d0b-d4a2330c968a
# ╟─4adb8b78-23f8-4d9b-9c34-78be3c460c8e
# ╠═ad153aef-42e0-4ed7-a81b-30f697007b0d
# ╠═459fa2ae-2a98-4a9c-8435-063270d1a7aa
# ╠═37fe2990-35ba-4b17-beae-235b141109b0
# ╟─9cbec465-48b2-42ca-9540-e2621ce0d8f3
# ╠═cdb20c28-e496-4013-9b30-5df057a12468
# ╠═50aff4f1-aebd-4af8-bf92-267c5fa07980
# ╠═b742cd3d-88f1-4740-8c29-6c29c5aacef9
# ╟─59402321-3b5f-4dbb-9cb4-888a6ee19bda
# ╟─7c8d245b-67b8-4968-bcdc-44119487d293
# ╠═c15d5f71-2c67-45b5-b9d3-9d93b8934ca7
# ╠═8a35e6e1-c166-4cde-aa95-d0db94512879
