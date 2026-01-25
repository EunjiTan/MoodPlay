import axios from 'axios';

const api = axios.create({
    baseURL: '/api' // Proxied by Vite to localhost:8000
});

export const uploadVideo = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post('/upload', formData);
    return res.data;
};

export const initSession = async (videoPath) => {
    const res = await api.post('/segment/init', null, { params: { video_path: videoPath } });
    return res.data;
};

export const sendClick = async (videoPath, frameIdx, objectId, points, labels) => {
    const res = await api.post('/segment/click', {
        video_path: videoPath,
        frame_idx: frameIdx,
        object_id: int(objectId),
        points: points, // [[x,y]]
        labels: labels  // [1]
    });
    return res.data;
};

export const generateVideo = async (videoPath, prompt) => {
    const res = await api.post('/generate', {
        video_path: videoPath,
        prompt: prompt
    });
    return res.data;
};

export default api;
