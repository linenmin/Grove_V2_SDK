import base64
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from grove.evidence import analyze
from grove.profiles import load_profile, package, validate_bundle, write_json, layout_check, digest
from grove.serial_io import capture, select_port


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profile = load_profile(ROOT/'deployments/profiles/optical-flow-observed-172x224.json')
        self.boot = ('1st BL Modem Build DATE=test\ncamera input init done\nEthos-U55 device initialised\ninitial done\n'
                     'model io: in(h=172,w=224,c=6) out(h=176,w=224,c=2)\n')

    def frame(self, size=(224,176), claimed=None):
        buf=io.BytesIO()
        Image.new('RGB',size).save(buf,format='JPEG')
        return json.dumps({'name':'INVOKE','code':0,'data':{'resolution':list(claimed or size),'image':base64.b64encode(buf.getvalue()).decode()}})+'\n'

    def run_log(self, content):
        log=self.root/'uart.bin'
        log.write_bytes(content.encode())
        return analyze(log,self.profile)

    def test_good_stream_and_timing(self):
        result=self.run_log(self.boot+self.frame()*5+'infer=109.2ms total=137.0ms\n')
        self.assertTrue(result['passed'])
        self.assertEqual(result['valid_frames'],5)
        self.assertFalse(result['identity_verified'])
        self.assertEqual(result['timing']['infer_ms_median'],109.2)

    def test_camera_fallback_rejected(self):
        self.assertFalse(self.run_log(self.boot+self.frame((320,240))*3)['passed'])

    def test_declared_resolution_cannot_hide_actual_shape(self):
        result=self.run_log(self.boot+self.frame((320,240),(224,176))*3)
        self.assertEqual(result['invalid_frames'],3)
        self.assertFalse(result['passed'])

    def test_old_success_cannot_hide_latest_failed_boot(self):
        result=self.run_log(self.boot+self.frame()*3+'1st BL Modem Build DATE=new\ncamera init fail\n')
        self.assertFalse(result['passed'])
        self.assertEqual(result['valid_frames'],0)

    def test_silent_partial_and_wrong_model_fail(self):
        self.assertFalse(self.run_log('')['passed'])
        self.assertFalse(self.run_log(self.boot+self.frame()*2+self.frame()[:40])['passed'])
        self.assertFalse(self.run_log(self.boot.replace('h=172','h=157')+self.frame()*3)['passed'])

    def test_observation_extracts_only_new_frames(self):
        log=self.root/'uart.bin'
        log.write_text(self.boot+self.frame()*8,encoding='utf-8')
        result=analyze(log,self.profile,self.root/'frames')
        self.assertEqual(result['valid_frames'],8)
        self.assertEqual(len(result['saved_frames']),3)
        with self.assertRaises(FileExistsError):
            analyze(log,self.profile,self.root/'frames')


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='grove bundle space ')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.profile=load_profile(ROOT/'deployments/profiles/optical-flow-157x203.json')
        self.model=self.root/'model with spaces.tflite'
        self.model.write_bytes(b'test artifact; not a real model')
        self.profile['models'][0].update(path=str(self.model),sha256=digest(self.model))
        self.firmware=self.root/'test.img'
        self.firmware.write_bytes(b'test firmware')

    def build(self):
        write_json(self.root/'profile.json',self.profile)
        package(self.root/'profile.json',self.firmware,self.root/'bundle')
        return self.root/'bundle/bundle.json'

    def test_bundle_integrity_and_no_overwrite(self):
        manifest=self.build()
        self.assertEqual(len(validate_bundle(manifest)['models']),1)
        (manifest.parent/'model-0.tflite').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            validate_bundle(manifest)
        with self.assertRaises(ValueError):
            package(self.root/'profile.json',self.firmware,self.root/'bundle')

    def test_observation_profile_cannot_package(self):
        self.profile['mode']='observe'
        with self.assertRaises(ValueError): self.build()

    def test_bad_model_hash_fails_before_creating_bundle(self):
        self.profile['models'][0]['sha256']='0'*64
        with self.assertRaises(ValueError): self.build()
        self.assertFalse((self.root/'bundle').exists())

    def test_overlap_capacity_and_offset(self):
        for address, offset in [('0x100','0x0'),('0xFFFFFF','0x0'),('0xB7B000','0x1')]:
            with self.subTest(address=address,offset=offset):
                with self.assertRaises(ValueError):
                    layout_check(self.profile,[{'address':address,'offset':offset,'bytes':100,'path':'test'}],10)

    def test_multiple_models_and_path_escape(self):
        self.profile['models'].append(dict(self.profile['models'][0],address='0xC00000'))
        manifest=self.build()
        bundle=validate_bundle(manifest)
        self.assertEqual(len(bundle['models']),2)
        bundle['firmware']['path']='../test.img'
        write_json(manifest,bundle)
        with self.assertRaises(ValueError): validate_bundle(manifest)


class SerialTests(unittest.TestCase):
    def test_ambiguous_device_not_selected(self):
        profile=load_profile(ROOT/'deployments/profiles/optical-flow-observed-172x224.json')
        entries=[{'vid':6790,'pid':21971,'port':'COM5','serial':'a'}, {'vid':6790,'pid':21971,'port':'COM6','serial':'b'}]
        with patch('grove.serial_io.ports', return_value=entries):
            with self.assertRaises(ValueError): select_port(profile)
            self.assertEqual(select_port(profile,serial_number='b')['port'],'COM6')

    def test_read_only_capture_stops_and_releases(self):
        class Fake:
            def open(self): self.closed=False
            def close(self): self.closed=True
            def read(self,n):
                (directory/'STOP').touch()
                return b'evidence'
            def write(self,data): raise AssertionError('monitor must not transmit')
        port=Fake()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            count=capture({'port':'fake'},921600,5,directory,lambda event:None,lambda:port)
            self.assertEqual(count,8)
            self.assertEqual((directory/'uart.bin').read_bytes(),b'evidence')
            self.assertTrue(port.closed)
            self.assertFalse(port.dtr)

    def test_read_error_releases_port(self):
        class Fake:
            def open(self): pass
            def read(self,n): raise OSError('unplugged')
            def close(self): self.closed=True
        port=Fake()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(OSError):
                capture({'port':'fake'},921600,1,Path(temp),lambda event:None,lambda:port)
        self.assertTrue(port.closed)


    def test_auto_reset_pulses_both_lines_after_listening(self):
        events=[]
        class Fake:
            def __setattr__(self,name,value):
                if name in ('rts','dtr'): events.append((name,value))
                object.__setattr__(self,name,value)
            def open(self): events.append('open')
            def close(self): pass
            def read(self,n):
                (directory/'STOP').touch()
                return b''
        with tempfile.TemporaryDirectory() as temp, patch('grove.serial_io.time.sleep'):
            directory=Path(temp)
            capture({'port':'fake'},921600,5,directory,lambda event:events.append(event['state']),Fake,auto_reset=True)
        start=events.index('listening')
        self.assertEqual(events[:events.index('open')],[('dtr',False),('rts',False)])
        self.assertEqual(events[start+1:start+6],[('rts',True),('dtr',True),('rts',False),('dtr',False),'reset_pulse_sent'])

    def test_flasher_wrapper_opens_low_and_pulses_once(self):
        spec=importlib.util.spec_from_file_location('xmodem_autoreset',ROOT/'tools/grove/xmodem_autoreset.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        events=[]
        class Fake:
            def __setattr__(self,name,value):
                if name in ('rts','dtr'): events.append((name,value))
                object.__setattr__(self,name,value)
            def open(self): events.append('open')
        module.install(Fake,delay=0)
        with patch('grove.serial_io.time.sleep'):
            port=Fake(); port.open()
            deadline=time.monotonic()+2
            while len(events)<7 and time.monotonic()<deadline: time.sleep(0.01)
            port.open()
        self.assertEqual(events[:7],[('dtr',False),('rts',False),'open',('rts',True),('dtr',True),('rts',False),('dtr',False)])
        self.assertEqual(events[7:],['open'])

class FlasherTests(unittest.TestCase):
    def test_deadline_and_spaces(self):
        spec=importlib.util.spec_from_file_location('grove_flasher',ROOT/'xmodem/xmodem_send.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.parse_model_arg('C:/space here/model.tflite 0xB7B000 0x0'),['C:/space here/model.tflite','0xB7B000','0x0'])
        module.deadline=-1
        with self.assertRaises(TimeoutError): module.bounded_readline()


if __name__ == '__main__': unittest.main()
