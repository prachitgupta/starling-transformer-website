import json,pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from index_videos import discover
class VideoTests(unittest.TestCase):
    def test_routing_escaping_and_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory); (root/'videos').mkdir(); (root/'data').mkdir()
            (root/'data/video_slots.json').write_text(json.dumps({'acp':{'keywords':['adaptive','coverage']},'expert':{'keywords':['expert','hrrt']}}))
            for name in ['acp_expert.mp4','HRRT flight.webm','unknown & clip.mp4','second adaptive.mp4']:
                (root/'videos'/name).touch()
            result=discover(root)
            self.assertEqual(len(result['acp']),2); self.assertEqual(len(result['expert']),1)
            self.assertEqual(result['additional'][0]['src'],'videos/unknown%20%26%20clip.mp4')
            (root/'videos/placements.json').write_text('{"unknown & clip.mp4":"expert"}')
            self.assertEqual(len(discover(root)['expert']),2)
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory); (root/'videos').mkdir(); (root/'data').mkdir()
            (root/'data/video_slots.json').write_text('{}'); self.assertEqual(discover(root),{'additional':[]})
    def test_invalid_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory); (root/'videos').mkdir(); (root/'data').mkdir()
            (root/'data/video_slots.json').write_text('{}'); (root/'videos/a.mp4').touch()
            (root/'videos/placements.json').write_text('{"a.mp4":"missing"}')
            with self.assertRaises(ValueError): discover(root)
if __name__=='__main__': unittest.main()
