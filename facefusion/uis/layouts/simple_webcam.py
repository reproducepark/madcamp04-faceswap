import gradio

from facefusion import state_manager
from facefusion.uis.components import about, source, webcam


def pre_check() -> bool:
	return True


def render() -> gradio.Blocks:
	# 고정된 설정들을 미리 설정
	state_manager.set_item('processors', ['face_swapper'])
	state_manager.set_item('face_swapper_model', 'hyperswap_1a_256')
	state_manager.set_item('face_swapper_pixel_boost', '256x256')
	state_manager.set_item('execution_providers', ['cuda', 'tensorrt'])
	state_manager.set_item('execution_thread_count', 16)
	state_manager.set_item('webcam_device_id', 0)
	state_manager.set_item('webcam_mode', 'udp')
	state_manager.set_item('webcam_resolution', '1280x720')
	state_manager.set_item('webcam_fps', 30)

	with gradio.Blocks(title="FaceFusion Simple Webcam") as layout:
		with gradio.Row():
			with gradio.Column(scale=4):
				with gradio.Blocks():
					about.render()
				
				# SOURCE만 선택 가능
				with gradio.Blocks():
					gradio.Markdown("### SOURCE 선택")
					source.render()
			
			with gradio.Column(scale=11):
				with gradio.Blocks():
					webcam.render()
	
	return layout


def listen() -> None:
	source.listen()
	webcam.listen()


def run(ui: gradio.Blocks) -> None:
	ui.launch(favicon_path='facefusion.ico', inbrowser=state_manager.get_item('open_browser')) 