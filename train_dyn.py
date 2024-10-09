import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from models import resnet
import numpy as np
from fullg_dyn_data import load_data
########################################################################################################################
#  Training Baseline
########################################################################################################################

parser = argparse.ArgumentParser(description='Trains ResNet on CIFAR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--lbd', type=float, default=0, help='lbd')
parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar100',choices=['cifar10', 'cifar100'],
                    help='Choose between Cifar10 and 100.')
parser.add_argument('--arch', type=str, default='resnet18')
# Optimization options
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')

# Checkpoints and Dynamics
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./checkpoint/all-dataset', help='Folder to save checkpoints and log.')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',default= False, help='evaluate model on validation set')
parser.add_argument('--dynamics', action='store_true', help='save training dynamics')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default='42', help='manual seed')

args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
device = args.device
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

def main():
    # Init logger
    print(args.save_path)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Dataset: {}".format(args.dataset), log)
    print_log("Data Path: {}".format(args.data_path), log)
    print_log("Network: {}".format(args.arch), log)
    print_log("Batchsize: {}".format(args.batch_size), log)
    print_log("Learning Rate: {}".format(args.learning_rate), log)
    print_log("Momentum: {}".format(args.momentum), log)
    print_log("Weight Decay: {}".format(args.decay), log)

    # data loading 
    train_data, train_loader, test_loader, trainloader1, trainloader2, trainloader3, trainloader4, trainloader5, trainloader11, trainloader22, trainloader33, trainloader44, trainloader55 = load_data(args)
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.num_samples = 50000
        args.num_iter = args.num_samples/args.batch_size
    if args.dataset == 'cifar100':
        args.num_classes = 100
        args.num_samples = 50000
        args.num_iter = args.num_samples/args.batch_size
    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    net = resnet.__dict__[args.arch](num_class = args.num_classes)
    net1 = resnet.__dict__[args.arch](num_class = args.num_classes)
    net2 = resnet.__dict__[args.arch](num_class = args.num_classes)
    net3 = resnet.__dict__[args.arch](num_class = args.num_classes)
    net4 = resnet.__dict__[args.arch](num_class = args.num_classes)
    net5 = resnet.__dict__[args.arch](num_class = args.num_classes)
    net_new = resnet.__dict__[args.arch](num_class = args.num_classes)
    print_log("=> network :\n {}".format(net), log)

    #net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    optimizer1 = torch.optim.SGD(net1.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    optimizer2 = torch.optim.SGD(net2.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    optimizer3 = torch.optim.SGD(net3.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    optimizer4 = torch.optim.SGD(net4.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    optimizer5 = torch.optim.SGD(net5.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    optimizer_new = torch.optim.SGD(net_new.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                        T_max =  args.epochs * args.num_iter)
    scheduler1 =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer1,
                                                        T_max =  args.epochs * args.num_iter)
    scheduler2 =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer2,
                                                        T_max =  args.epochs * args.num_iter)
    scheduler3 =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer3,
                                                        T_max =  args.epochs * args.num_iter)
    scheduler4 =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer4,
                                                        T_max =  args.epochs * args.num_iter)
    scheduler5 =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer5,
                                                        T_max =  args.epochs * args.num_iter)
    
    if args.use_cuda:
        net.to(device)
        net1.to(device)
        net2.to(device)
        net3.to(device)
        net4.to(device)
        net5.to(device)
        net_new.to(device)
        criterion.to(device)

    recorder = RecorderMeter(args.epochs)
    # evaluation
    if args.evaluate:
        time1 = time.time()
        validate(test_loader, args, net, criterion, log) #
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    
    print("model_all train")
    for epoch in range(args.epochs):

        current_learning_rate = scheduler.get_last_lr()[0]

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los, loss_epoch, output_epoch, index_epoch = train(train_loader, args, net, criterion, optimizer, scheduler, epoch, log)
        '''
        print("loss_epoch:",loss_epoch)
        print("output_epoch:",output_epoch)
        print("index_epoch:",index_epoch)
        print("len(loss_epoch):",len(loss_epoch))
        print("len(output_epoch):",len(output_epoch))
        print("len(index_epoch):",len(index_epoch))'''
        # evaluate on validation set
        val_acc, val_los = validate(test_loader, args, net, criterion, log)

        is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        '''
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net,
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_path, 'checkpoint.pth.tar')
        '''
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))
        # save training dynamics
        '''
        if args.dynamics:
            dynamics_path = args.save_path+'/npy/'
            if not os.path.exists(dynamics_path):
                os.makedirs(dynamics_path)
            np.save(args.save_path+'/npy/'+ str(epoch) + '_Loss.npy', loss_epoch)
            np.save(args.save_path+'/npy/'+ str(epoch) + '_Output.npy', output_epoch)
            np.save(args.save_path+'/npy/'+ str(epoch) + '_Index.npy', index_epoch)
            print('Epoch '+str(epoch)+' done!')'''

    print("mul_model_train")
    for epoch in range(args.epochs):

        current_learning_rate = scheduler1.get_last_lr()[0]

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc1, train_los1, loss_epoch1, output_epoch1, index_epoch1 = xtrain(trainloader11, trainloader1, args, net, net1, criterion, optimizer1, scheduler1, epoch, log)
        train_acc2, train_los2, loss_epoch2, output_epoch2, index_epoch2 = xtrain(trainloader22, trainloader2, args, net, net2, criterion, optimizer2, scheduler2, epoch, log)
        train_acc3, train_los3, loss_epoch3, output_epoch3, index_epoch3 = xtrain(trainloader33, trainloader3, args, net, net3, criterion, optimizer3, scheduler3, epoch, log)
        train_acc4, train_los4, loss_epoch4, output_epoch4, index_epoch4 = xtrain(trainloader44, trainloader4, args, net, net4, criterion, optimizer4, scheduler4, epoch, log)
        train_acc5, train_los5, loss_epoch5, output_epoch5, index_epoch5 = xtrain(trainloader55, trainloader5, args, net, net5, criterion, optimizer5, scheduler5, epoch, log)


        output_epoch = np.concatenate([output_epoch1, output_epoch2, output_epoch3, output_epoch4, output_epoch5])

        loss_epoch = np.concatenate((loss_epoch1,loss_epoch2,loss_epoch3,loss_epoch4,loss_epoch5),axis=0)
        output_epoch = np.concatenate((output_epoch1,output_epoch2,output_epoch3,output_epoch4,output_epoch5),axis=0)
        index_epoch = np.concatenate((index_epoch1,index_epoch2,index_epoch3,index_epoch4,index_epoch5),axis=0)
        '''
        print("loss_epoch:",loss_epoch)
        print("output_epoch:",output_epoch)
        print("index_epoch:",index_epoch)
        print("len(loss_epoch):",len(loss_epoch))
        print("len(output_epoch):",len(output_epoch))
        print("len(index_epoch):",len(index_epoch))'''
        # evaluate on validation set
        val_acc, val_los = validate(test_loader, args, net1, criterion, log)

        is_best = recorder.update(epoch, train_los1, train_acc1, val_los, val_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net,
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_path, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))
        # save training dynamics
        if args.dynamics:
            dynamics_path = args.save_path+'/npy/'
            if not os.path.exists(dynamics_path):
                os.makedirs(dynamics_path)
            np.save(args.save_path+'/npy/'+ str(epoch) + '_Loss.npy', loss_epoch)
            np.save(args.save_path+'/npy/'+ str(epoch) + '_Output.npy', output_epoch)
            np.save(args.save_path+'/npy/'+ str(epoch) + '_Index.npy', index_epoch)
            print('Epoch '+str(epoch)+' done!')
    log.close()

# train function (forward, backward, update)
def train(train_loader, args, model, criterion, optimizer, scheduler, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    
    for t, (input, target) in enumerate(train_loader):
        if args.use_cuda:
            y = target[0].to(device)
            x = input.to(device)
            index = target[1]
        
        input_var = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(y)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # record training dynamics
        loss_batch = torch.nn.functional.cross_entropy(output, target_var, reduce=False).detach().cpu()
        index_batch = index
        if t==0:
            loss_epoch = np.array(loss_batch)
            output_epoch = np.array(output.detach().cpu())
            index_epoch = np.array(index_batch)
        else:
            loss_epoch = np.concatenate((loss_epoch, np.array(loss_batch)), axis = 0)
            output_epoch = np.concatenate((output_epoch, np.array(output.detach().cpu())), axis = 0)
            index_epoch = np.concatenate((index_epoch, np.array(index_batch)), axis = 0)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
        losses.update(loss.item(), len(y))
        top1.update(prec1.item(), len(y))
        top5.update(prec5.item(), len(y))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if t % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, t, args.batch_size, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg), log)
    return top1.avg, losses.avg, loss_epoch, output_epoch, index_epoch


def xtrain(trainloader11, trainloader1, args, model_all, model, criterion, optimizer, scheduler, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    lbd = args.lbd
    for t, (input, target) in enumerate(trainloader11):
        if args.use_cuda:
            y1 = target[0].to(device)
            x1 = input.to(device)
        
        input_var1 = torch.autograd.Variable(x1)
        target_var1 = torch.autograd.Variable(y1)
        # compute output
        output1 = model(input_var1)
        loss1 = criterion(output1, target_var1)
        
        optimizer.zero_grad()
        loss1.backward()

        optimizer.step()
        scheduler.step()

    for t, (input, target) in enumerate(trainloader1):
        if args.use_cuda:
            y = target[0].to(device)
            x = input.to(device)
            index = target[1]
        
        input_var = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(y)
        # compute output
        output_all = model_all(input_var)
        output = model(input_var)
        loss = criterion(output, target_var)
        # record training dynamics
        loss_all = torch.nn.functional.cross_entropy(output_all, target_var, reduce=False).detach().cpu()
        loss_1 = torch.nn.functional.cross_entropy(output, target_var, reduce=False).detach().cpu()
        loss_batch = loss_all + lbd * loss_1
        index_batch = index
        if t==0:
            loss_epoch = np.array(loss_batch)
            output_epoch = np.array(output.detach().cpu())
            index_epoch = np.array(index_batch)
        else:
            loss_epoch = np.concatenate((loss_epoch, np.array(loss_batch)), axis = 0)
            output_epoch = np.concatenate((output_epoch, np.array(output.detach().cpu())), axis = 0)
            index_epoch = np.concatenate((index_epoch, np.array(index_batch)), axis = 0)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
        losses.update(loss.item(), len(y))
        top1.update(prec1.item(), len(y))
        top5.update(prec5.item(), len(y))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if t % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, t, args.batch_size, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg), log)
    return top1.avg, losses.avg, loss_epoch, output_epoch, index_epoch

def validate(test_loader, args, model, criterion, log): 
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():

        for i, (input, target) in enumerate(test_loader):
            if args.use_cuda:
                y = target.to(device)
                x = input.to(device)

            # compute output
            output = model(x)
            loss = criterion(output, y)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
            losses.update(loss.item(), len(y))
            top1.update(prec1.item(), len(y))
            top5.update(prec5.item(), len(y))

        print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                    error1=100 - top1.avg),
                log)

    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
