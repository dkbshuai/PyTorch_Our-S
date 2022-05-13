# helper methods
    
def exists(val):
    return val is not None

# classes

class DistillWrapper(nn.Module):
    def __init__(
        self,
        *,
        teacher,
        temperature = 5,
        alpha = 0.5,
    ):
        super(DistillWrapper, self ).__init__()
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, img, labels, student_logits, temperature = None, alpha = None, **kwargs):
        b, *_ = img.shape
        alpha = alpha if exists(alpha) else self.alpha
        T = temperature if exists(temperature) else self.temperature

        with torch.no_grad():
            teacher_logits = self.teacher(img)
            
        loss = F.cross_entropy(student_logits, labels)

        soft_distill_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim = -1),
            F.softmax(teacher_logits / T, dim = -1).detach(),
            reduction = 'batchmean')
        soft_distill_loss *= T ** 2
            
        teacher_labels = teacher_logits.argmax(dim = -1)
        hard_distill_loss = F.cross_entropy(student_logits, teacher_labels)

        return loss + soft_distill_loss * alpha + hard_distill_loss * (1 - alpha)
