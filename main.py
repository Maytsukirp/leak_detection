from modules.optical_flow import OpticalFlow

def run():
    path = 'resources/4_GHG_videos_pack_4/MOV_4318.mp4'
    optical_flow = OpticalFlow(path)
    optical_flow.optical_flow_gf()

if __name__ == "__main__":
    run()
    
    
