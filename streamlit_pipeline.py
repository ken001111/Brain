import streamlit as st
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import tempfile
import os
import subprocess
from pathlib import Path
import shutil

# PyTorch 임포트 (선택적)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch가 설치되어 있지 않습니다. 모델 추론 기능을 사용할 수 없습니다.")

# 페이지 설정
st.set_page_config(
    page_title="CT Eye Detection & Alignment",
    page_icon="👁️",
    layout="wide"
)

# 세션 상태 초기화
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

def create_temp_directory():
    """임시 디렉토리 생성"""
    temp_dir = tempfile.mkdtemp()
    return temp_dir

def cleanup_temp_directory(temp_dir):
    """임시 디렉토리 정리"""
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def load_model_and_predict(ct_path, model_path):
    """직접 모델을 로드하여 추론 수행"""
    if not TORCH_AVAILABLE:
        st.error("PyTorch가 설치되어 있지 않습니다!")
        return None, None, None
        
    import torch
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError:
        st.error("nnUNetv2가 설치되어 있지 않습니다!")
        return None, None, None
    
    # GPU 사용 가능 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.info(f"사용 중인 디바이스: {device}")
    
    try:
        # 모델 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=device)
        
        # nnUNet predictor 초기화
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=True if device.type == 'cuda' else False,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        
        # 모델 파라미터 설정
        predictor.initialize_from_trained_model_folder(
            model_folder_path=os.path.dirname(model_path),
            use_folds=None,
            checkpoint_name=os.path.basename(model_path)
        )
        
        # CT 데이터 로드
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata()
        
        # 입력 데이터 준비 (nnUNet 형식)
        ct_data_input = ct_data[np.newaxis, ...]  # 채널 차원 추가
        
        # 추론 수행
        with torch.no_grad():
            prediction = predictor.predict_from_data_iterator(
                data_iterator=[(ct_data_input, {'spacing': ct_nii.header.get_zooms()})],
                save_probabilities=False,
                num_processes_segmentation_export=1
            )
        
        # 결과 추출
        mask_data = prediction[0][1]  # segmentation mask
        
        return ct_data, mask_data, ct_nii
        
    except Exception as e:
        st.error(f"모델 로드 또는 추론 중 오류: {str(e)}")
        return None, None, None


def simple_inference_with_torch(ct_path, model_path):
    """단순화된 PyTorch 직접 추론"""
    if not TORCH_AVAILABLE:
        st.error("PyTorch가 설치되어 있지 않습니다!")
        return None, None, None
        
    import torch
    import torch.nn.functional as F
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # CT 데이터 로드
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata()
        
        # 정규화 (nnUNet 스타일)
        ct_data = np.clip(ct_data, -1024, 3071)  # HU 범위 제한
        mean = np.mean(ct_data)
        std = np.std(ct_data)
        ct_data_norm = (ct_data - mean) / (std + 1e-8)
        
        # 텐서로 변환
        ct_tensor = torch.from_numpy(ct_data_norm).float()
        ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        ct_tensor = ct_tensor.to(device)
        
        # 모델 로드
        model = torch.load(model_path, map_location=device)
        if isinstance(model, dict):
            # state_dict인 경우
            st.warning("모델 아키텍처가 필요합니다. nnUNet 모델 폴더 경로를 사용해주세요.")
            return None, None, None
        
        model.eval()
        
        # 추론
        with torch.no_grad():
            output = model(ct_tensor)
            
            # Softmax 적용 (다중 클래스인 경우)
            if output.shape[1] > 1:
                output = F.softmax(output, dim=1)
                mask = torch.argmax(output, dim=1)
            else:
                mask = torch.sigmoid(output) > 0.5
        
        # numpy로 변환
        mask_data = mask.squeeze().cpu().numpy()
        
        return ct_data, mask_data, ct_nii
        
    except Exception as e:
        st.error(f"추론 중 오류: {str(e)}")
        return None, None, None

def find_eye_centers(mask_data):
    """마스크에서 눈 중심점 찾기"""
    labeled_3d = label(mask_data > 0)
    regions_3d = regionprops(labeled_3d)
    
    centers = []
    for region in regions_3d:
        z_center, y_center, x_center = region.centroid
        centers.append((x_center, y_center))
    
    centers_sorted = sorted(centers, key=lambda c: c[0])
    return centers_sorted, len(regions_3d)

def calculate_rotation_angle(centers):
    """두 중심점의 기울기로 회전 각도 계산"""
    if len(centers) < 2:
        return 0, "단일 영역만 검출됨"
    
    left_eye = centers[0]
    right_eye = centers[1]
    
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    return -angle_deg, None

def rotate_volume(volume_data, angle_deg):
    """볼륨 데이터 회전"""
    rotated = ndimage.rotate(
        volume_data,
        angle_deg,
        axes=(1, 2),
        reshape=False,
        order=1,
        mode='constant',
        cval=volume_data.min()
    )
    return rotated

def create_visualization(ct_data, mask_data, rotated_ct, rotated_mask, angle):
    """시각화 생성"""
    mask_slices = np.where(mask_data.sum(axis=(1,2)) > 0)[0]
    
    if len(mask_slices) == 0:
        return None
    
    # 중간 슬라이스 선택
    center_slice = mask_slices[len(mask_slices)//2]
    
    fig, axes = plt.subplots

# create_visualization 함수 계속
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 원본 CT
    axes[0].imshow(ct_data[center_slice, :, :], cmap='gray')
    axes[0].set_title('Original CT')
    axes[0].axis('off')
    
    # 원본 CT + Mask
    axes[1].imshow(ct_data[center_slice, :, :], cmap='gray')
    axes[1].imshow(mask_data[center_slice, :, :], cmap='Reds', alpha=0.5)
    axes[1].set_title('Original + Mask')
    axes[1].axis('off')
    
    # 회전된 CT
    axes[2].imshow(rotated_ct[center_slice, :, :], cmap='gray')
    axes[2].set_title(f'Rotated CT ({angle:.1f}°)')
    axes[2].axis('off')
    
    # 회전된 CT + Mask
    axes[3].imshow(rotated_ct[center_slice, :, :], cmap='gray')
    axes[3].imshow(rotated_mask[center_slice, :, :], cmap='Reds', alpha=0.5)
    axes[3].set_title('Rotated + Mask')
    axes[3].axis('off')
    
    plt.tight_layout()
    return fig

# Streamlit UI
st.title("🧠 CT Eye Detection & Alignment Pipeline")
st.markdown("---")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    inference_method = st.radio(
        "추론 방법",
        ["PyTorch 모델 파일 (.pth)", "nnUNet 모델 폴더"],
        index=0
    )
    
    if inference_method == "PyTorch 모델 파일 (.pth)":
        model_file = st.file_uploader(
            "모델 파일 업로드",
            type=['pth', 'pt'],
            help="학습된 best_model.pth 파일을 업로드하세요"
        )
        
        if model_file:
            # 모델 파일을 임시 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                tmp_file.write(model_file.getbuffer())
                model_path = tmp_file.name
                st.session_state.model_path = model_path
                st.success(f"✅ 모델 로드 완료: {model_file.name}")
    else:
        model_path = st.text_input(
            "nnUNet 모델 폴더 경로",
            value="nnUNet_results/Dataset001_Eyeball/nnUNetTrainer__nnUNetPlans__3d_fullres",
            help="학습된 nnUNet 모델이 있는 폴더 경로"
        )
        st.session_state.model_path = model_path
    
    st.markdown("---")
    
    # 추론 설정
    st.subheader("🔧 추론 설정")
    
    use_gpu = st.checkbox("GPU 사용", value=TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False)
    
    if use_gpu and TORCH_AVAILABLE and not torch.cuda.is_available():
        st.warning("⚠️ GPU를 사용할 수 없습니다. CPU로 실행됩니다.")
    elif not TORCH_AVAILABLE:
        st.error("⚠️ PyTorch가 설치되어 있지 않습니다.")
    
    batch_size = st.number_input("배치 크기", min_value=1, max_value=8, value=1)
    
    st.markdown("---")
    st.markdown("### 📝 사용 방법")
    st.markdown("""
    1. 모델 파일(.pth) 또는 경로 설정
    2. CT 파일(.nii.gz) 업로드
    3. 'Process' 버튼 클릭
    4. 결과 확인 및 다운로드
    """)
    
    # PyTorch 임포트 확인
    if TORCH_AVAILABLE:
        st.sidebar.success(f"PyTorch {torch.__version__} 사용 가능")
        if torch.cuda.is_available():
            st.sidebar.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.error("PyTorch가 설치되어 있지 않습니다!")
        st.sidebar.info("설치: pip install torch torchvision")

# 메인 컨텐츠
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📤 파일 업로드")
    uploaded_file = st.file_uploader(
        "CT 파일 선택 (.nii.gz)",
        type=['nii.gz'],
        help="NIfTI 형식의 CT 스캔 파일을 업로드하세요"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ 업로드 완료: {uploaded_file.name}")
        
        # 파일 정보 표시
        file_size = uploaded_file.size / (1024 * 1024)  # MB
        st.info(f"파일 크기: {file_size:.2f} MB")

with col2:
    st.header("🔄 처리 상태")
    
    if uploaded_file is not None:
        if st.button("🚀 Process", type="primary", use_container_width=True):
            st.session_state.processed = False
            
            # 진행 상태 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 임시 디렉토리 생성
                if st.session_state.temp_dir:
                    cleanup_temp_directory(st.session_state.temp_dir)
                
                temp_dir = create_temp_directory()
                st.session_state.temp_dir = temp_dir
                
                # 1. 파일 저장
                status_text.text("📁 파일 저장 중...")
                progress_bar.progress(10)
                
                input_dir = os.path.join(temp_dir, "input")
                os.makedirs(input_dir, exist_ok=True)
                
                input_path = os.path.join(input_dir, uploaded_file.name)
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. 모델 추론
                status_text.text("🧠 AI 모델로 눈 검출 중...")
                progress_bar.progress(30)
                
                # 모델이 업로드되었는지 확인
                if 'model_path' not in st.session_state:
                    st.error("❌ 모델을 먼저 업로드해주세요!")
                    progress_bar.empty()
                    status_text.empty()
                    st.stop()
                
                # 실제 모델 추론
                if inference_method == "PyTorch 모델 파일 (.pth)":
                    # 간단한 추론 (모델 구조를 모르는 경우)
                    ct_data, mask_data, ct_nii = simple_inference_with_torch(
                        input_path, 
                        st.session_state.model_path
                    )
                else:
                    # nnUNet 스타일 추론
                    ct_data, mask_data, ct_nii = load_model_and_predict(
                        input_path,
                        st.session_state.model_path
                    )
                
                # 추론 실패 시 데모 데이터 사용
                if ct_data is None or mask_data is None:
                    st.warning("⚠️ 모델 추론 실패. 데모 데이터를 사용합니다.")
                    
                    ct_nii = nib.load(input_path)
                    ct_data = ct_nii.get_fdata()
                    
                    # 가상의 마스크 생성 (실제로는 모델 출력 사용)
                    mask_data = np.zeros_like(ct_data)
                    center_z = ct_data.shape[0] // 2
                    for x_offset in [-60, 60]:
                        for z in range(max(0, center_z-5), min(ct_data.shape[0], center_z+5)):
                            for y in range(200, 280):
                                for x in range(256+x_offset-30, 256+x_offset+30):
                                    if ((y-240)**2 + (x-256-x_offset)**2) < 900:
                                        mask_data[z, y, x] = 1
                
                progress_bar.progress(60)
                
                # 3. 중심점 찾기 및 각도 계산
                status_text.text("📐 정렬 각도 계산 중...")
                progress_bar.progress(70)
                
                centers, num_regions = find_eye_centers(mask_data)
                angle, error_msg = calculate_rotation_angle(centers)
                
                if error_msg:
                    st.warning(f"⚠️ {error_msg}")
                    angle = 0
                
                # 4. 회전 수행
                status_text.text("🔄 이미지 회전 중...")
                progress_bar.progress(80)
                
                if abs(angle) > 0.5:
                    rotated_ct = rotate_volume(ct_data, angle)
                    rotated_mask = rotate_volume(mask_data, angle)
                else:
                    rotated_ct = ct_data
                    rotated_mask = mask_data
                    st.info("회전 각도가 0.5도 미만이어서 회전하지 않았습니다.")
                
                # 5. 결과 저장
                status_text.text("💾 결과 저장 중...")
                progress_bar.progress(90)
                
                # 결과 파일 저장
                result_ct_path = os.path.join(temp_dir, "aligned_ct.nii.gz")
                result_mask_path = os.path.join(temp_dir, "aligned_mask.nii.gz")
                
                rotated_ct_nii = nib.Nifti1Image(rotated_ct, ct_nii.affine, ct_nii.header)
                rotated_mask_nii = nib.Nifti1Image(rotated_mask, ct_nii.affine, ct_nii.header)
                
                nib.save(rotated_ct_nii, result_ct_path)
                nib.save(rotated_mask_nii, result_mask_path)
                
                # 세션 상태 업데이트
                st.session_state.processed = True
                st.session_state.results = {
                    'ct_data': ct_data,
                    'mask_data': mask_data,
                    'rotated_ct': rotated_ct,
                    'rotated_mask': rotated_mask,
                    'angle': angle,
                    'num_regions': num_regions,
                    'centers': centers,
                    'result_ct_path': result_ct_path,
                    'result_mask_path': result_mask_path
                }
                
                progress_bar.progress(100)
                status_text.text("✅ 처리 완료!")
                
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                progress_bar.empty()
                status_text.empty()
# 결과 표시
if st.session_state.processed and 'results' in st.session_state:
    st.markdown("---")
    st.header("📊 결과")
    
    results = st.session_state.results
    
    # 통계 정보
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("검출된 영역 수", results['num_regions'])
    with col2:
        st.metric("회전 각도", f"{results['angle']:.2f}°")
    with col3:
        st.metric("중심점 수", len(results['centers']))
    
    # 시각화
    st.subheader("🖼️ 시각화")
    fig = create_visualization(
        results['ct_data'],
        results['mask_data'],
        results['rotated_ct'],
        results['rotated_mask'],
        results['angle']
    )
    
    if fig:
        st.pyplot(fig)
        plt.close()
    
    # 다운로드 버튼
    st.subheader("💾 다운로드")
    col1, col2 = st.columns(2)
    
    with col1:
        with open(results['result_ct_path'], 'rb') as f:
            st.download_button(
                label="📥 정렬된 CT 다운로드",
                data=f.read(),
                file_name="aligned_ct.nii.gz",
                mime="application/gzip"
            )
    
    with col2:
        with open(results['result_mask_path'], 'rb') as f:
            st.download_button(
                label="📥 정렬된 마스크 다운로드",
                data=f.read(),
                file_name="aligned_mask.nii.gz",
                mime="application/gzip"
            )

# 정리
if st.button("🗑️ 임시 파일 정리"):
    if st.session_state.temp_dir:
        cleanup_temp_directory(st.session_state.temp_dir)
        st.session_state.temp_dir = None
        st.session_state.processed = False
        st.success("✅ 임시 파일이 정리되었습니다.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>CT Eye Detection & Alignment Pipeline v1.0</p>
    </div>
    """,
    unsafe_allow_html=True
)