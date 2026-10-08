# -*- coding: utf-8 -*-
"""
GPU Status Monitor for AI Vision Training Platform
"""

class GPUMonitor:
    @staticmethod
    def get_gpu_info():
        info = {
            "cuda_available": False,
            "cuda_version": "N/A",
            "device_count": 0,
            "devices": []
        }

        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
            info["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0

            if info["cuda_available"]:
                for i in range(info["device_count"]):
                    prop = torch.cuda.get_device_properties(i)
                    total_mem_gb = round(prop.total_memory / (1024 ** 3), 2)
                    
                    allocated_mem_gb = 0.0
                    try:
                        allocated_mem_gb = round(torch.cuda.memory_allocated(i) / (1024 ** 3), 2)
                    except Exception:
                        pass
                    
                    info["devices"].append({
                        "id": i,
                        "name": prop.name,
                        "total_memory_gb": total_mem_gb,
                        "used_memory_gb": allocated_mem_gb,
                        "free_memory_gb": round(total_mem_gb - allocated_mem_gb, 2),
                        "multi_processor_count": prop.multi_processor_count
                    })
        except Exception as e:
            pass

        return info
